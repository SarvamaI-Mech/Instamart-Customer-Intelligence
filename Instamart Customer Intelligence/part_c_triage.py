"""Part C -- support-ticket triage with guardrails, redaction, retries, output validation, budget and routing.

Run:  python part_c_triage.py        (needs outputs/customer_segments.csv from Part B)
Writes outputs/triage_budget_demo.csv, outputs/triage_results.csv, outputs/triage_retry_log.txt
"""
import json
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
ALLOWED = ["billing", "delivery", "technical", "fraud"]

# redaction rules: applied in exactly this order
REDACTIONS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT = ('You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. '
             'Reply with JSON only, no other text: {"category": "...", "reason": "..."}')
FEW_SHOT = ZERO_SHOT + '''
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "duplicate charge"}
Ticket: The rider never arrived with [ORDER_ID]. -> {"category": "delivery", "reason": "order not received"}
Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}
Ticket: Someone used my wallet without permission, this is fraud. -> {"category": "fraud", "reason": "unauthorized use"}'''

RETRY_LOG: list = []


# --------------------------------------------------------------------------- redaction
def redact(text: str, counts: dict | None = None) -> str:
    for name, pattern, placeholder in REDACTIONS:
        text, n = pattern.subn(placeholder, text)
        if counts is not None:
            counts[name] = counts.get(name, 0) + n
    return text


# --------------------------------------------------------------------------- resilient call
def call_with_retry(messages: list, ticket_id: str, stats: dict):
    """Call the model; on TransientLLMError wait BASE_DELAY * 2**n * U(0.9, 1.15) and retry (max 3 retries)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient_errors"] += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            RETRY_LOG.append(f"retry ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s")
            time.sleep(wait)


# --------------------------------------------------------------------------- output validation
def validate_reply(text: str):
    """Return (status, category). Never guesses: ambiguous or off-list answers go to human review."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("category") in ALLOWED:
            return "classified_json", obj["category"]
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        pass
    found = {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w in ALLOWED}
    if len(found) == 1:
        return "classified_keyword", next(iter(found))
    return "flagged_invalid_output", None


# --------------------------------------------------------------------------- pipeline
def run_pipeline(tickets: pd.DataFrame, system_prompt: str, token_budget: int | None = None):
    """Process tickets in file order. Returns (per-ticket DataFrame, stats dict)."""
    mock_llm.reset()
    stats = {"transient_errors": 0, "model_calls": 0, "total_tokens": 0, "prompts_checked": 0}
    rows = []
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": None, "category": "", "tokens_used": 0}
        text = t.text
        if not text.strip():                                         # guardrail (a)
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:                    # guardrail (b)
            row["status"] = "flagged_long"
        elif token_budget is not None and stats["total_tokens"] >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            clean = redact(text)
            assert not any(p.search(clean) for _, p, _ in REDACTIONS), "PII reached the prompt"
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            stats["prompts_checked"] += 1
            resp = call_with_retry(messages, t.ticket_id, stats)
            stats["model_calls"] += 1
            used = resp.usage.total_tokens
            stats["total_tokens"] += used
            row["tokens_used"] = used
            row["status"], cat = validate_reply(resp.choices[0].message.content)
            row["category"] = cat or ""
        rows.append(row)
    return pd.DataFrame(rows), stats


def evaluate(res: pd.DataFrame, tickets: pd.DataFrame, stats: dict, label: str) -> dict:
    merged = res.merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
    ok = merged[merged["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((ok["category"] == ok["true_category"]).sum())
    out = {"label": label, "classified": len(ok), "correct": correct, "accuracy": correct / len(ok),
           "total_tokens": stats["total_tokens"], "tokens_per_classified": stats["total_tokens"] / len(ok),
           "status_counts": res["status"].value_counts().to_dict(), "transient_errors": stats["transient_errors"]}
    return out, ok


# --------------------------------------------------------------------------- routing
def add_routing(res: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
    out = res.copy()
    out["segment_name"] = out["customer_id"].map(seg["segment_name"]).fillna("")
    premium = out["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)   # unknown customer -> not premium
    human = out["status"].isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    fraud = (~human) & (out["category"] == "fraud")
    prem = (~human) & (~fraud) & premium
    out["route"] = "support_queue"; out["priority"] = "NORMAL"
    out.loc[human, "route"] = "human_review"
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[prem, ["route", "priority"]] = ["premium_desk", "HIGH"]
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


def main() -> None:
    random.seed(42)
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    tok = tickets["text"].map(est_tokens)
    print(f"tickets={len(tickets)} empty={(~tickets['text'].str.strip().astype(bool)).sum()} "
          f"long(>{MAX_INPUT_TOKENS} tokens)={(tok > MAX_INPUT_TOKENS).sum()} long est_tokens={sorted(set(tok[tok > MAX_INPUT_TOKENS]))}")

    # redaction report over all 164 tickets
    counts: dict = {}
    redacted = tickets["text"].map(lambda s: redact(s, counts))
    left = sum(len(p.findall(s)) for s in redacted for _, p, _ in REDACTIONS)
    print(f"substitutions: {counts} total={sum(counts.values())}; patterns still found after redaction: {left}")
    assert left == 0

    # two prompts, full runs without budget
    reports = {}
    for label, prompt in (("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)):
        assert "CLASSIFY" in prompt and all(c in prompt for c in ALLOWED)
        assert ("EXAMPLES:" in prompt) == (label == "few-shot")
        RETRY_LOG.clear()
        res, stats = run_pipeline(tickets, prompt)
        rep, ok = evaluate(res, tickets, stats, label)
        reports[label] = rep
        print(f"\n=== {label} === prompt est_tokens={est_tokens(prompt)}")
        print(f"status counts: {rep['status_counts']}")
        print(f"TransientLLMErrors={stats['transient_errors']} retry log lines={len(RETRY_LOG)} prompts PII-checked={stats['prompts_checked']}")
        print(f"classified={rep['classified']} correct={rep['correct']} accuracy={rep['accuracy']:.1%} "
              f"total_tokens={rep['total_tokens']} tokens/classified={rep['tokens_per_classified']:.1f}")
        waits = [float(l.split("wait=")[1].rstrip("s")) for l in RETRY_LOG]
        assert all(0.045 <= w <= 0.0575 for w in waits), "wait outside jitter range for retry 0"
        if label == "zero-shot":
            print("confusion (rows=true, cols=predicted):")
            truth = ok["ticket_id"].map(tickets.set_index("ticket_id")["true_category"])
            print(pd.crosstab(truth, ok["category"], rownames=["true"], colnames=["predicted"]).to_string())
            with open("outputs/triage_retry_log.txt", "w") as f:
                f.write("\n".join(RETRY_LOG) + "\n")
            print("first retry log lines:", *RETRY_LOG[:3], sep="\n  ")

    # budget demo (zero-shot)
    demo_budget = 6000
    res, stats = run_pipeline(tickets, ZERO_SHOT, token_budget=demo_budget)
    n_ex = int((res["status"] == "budget_exhausted").sum())
    if n_ex == 0:
        demo_budget = 5000
        res, stats = run_pipeline(tickets, ZERO_SHOT, token_budget=demo_budget)
        n_ex = int((res["status"] == "budget_exhausted").sum())
    add_routing(res, tickets).to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"\nBudget demo: TOKEN_BUDGET={demo_budget}; budget_exhausted tickets={n_ex}; tokens spent={stats['total_tokens']}")
    assert n_ex >= 1

    # final routed results (few-shot, budget 100000)
    res, stats = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    final = add_routing(res, tickets)
    final.to_csv("outputs/triage_results.csv", index=False)
    print(f"\nFinal triage rows={len(final)}")
    print(final.groupby(["route", "priority"]).size().to_string())
    print("status counts:", final["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
