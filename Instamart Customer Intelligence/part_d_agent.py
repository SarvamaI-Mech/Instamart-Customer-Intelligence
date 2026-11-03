"""Part D (tasks 5-8) -- refund agent: tools, ReAct loop, safety limits, checkpoint/resume.

Run:  python part_d_agent.py
Writes outputs/agent_runs.jsonl (the 7 scenarios), outputs/agent_demos.jsonl (stop-safely demos),
outputs/refund_ledger.jsonl and checkpoints in state/.
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

import mock_llm
from part_d_rag import retrieve

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
STATE_DIR = "state"
RUNS, DEMOS = "outputs/agent_runs.jsonl", "outputs/agent_demos.jsonl"
ESCALATION_LIMIT_INR = 1000
LATE_MINUTES = 45

AGENT_SYSTEM = ("You are a refund AGENT. Use the tools to look up the order, check the policy, calculate the refund "
                "and then issue it or escalate it. Finish with STATUS: explanation.")


# --------------------------------------------------------------------------- tools
def get_order(order_id: str) -> dict:
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    oid, status, amount, minutes = row
    return {"order_id": oid, "status": status, "order_amount": float(amount),
            "delivery_minutes": None if minutes is None or pd.isna(minutes) else float(minutes)}


def search_policy(query: str) -> dict:
    return {"results": retrieve(query, 3)}


def calculate_refund(order_amount: float, delivery_minutes) -> dict:
    """Full refund when delivered in more than 45 minutes, otherwise 0. (Order status is checked by the agent.)"""
    late = delivery_minutes is not None and float(delivery_minutes) > LATE_MINUTES
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def _ledger(tool: str, order_id: str, refund_inr: float) -> dict:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
             "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id: str, refund_inr: float) -> dict:
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict:
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, description, props):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the database.",
          {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the three best-matching passages.",
          {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund (up to INR 1000) and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# --------------------------------------------------------------------------- checkpoints
class SimulatedCrash(RuntimeError):
    pass


def _ckpt_path(run_id: str) -> str:
    return os.path.join(STATE_DIR, f"{run_id}.json")


def save_checkpoint(st: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    data = {k: st[k] for k in ("run_id", "scenario_id", "user_message", "messages", "tokens_used", "step",
                               "llm_calls", "tools_executed", "steps", "max_iterations", "token_limit",
                               "time_limit_seconds")}
    with open(_ckpt_path(st["run_id"]), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)


# --------------------------------------------------------------------------- ReAct loop
def _fallback(st: dict, status: str, reason: str) -> dict:
    tools = ", ".join(st["tools_executed"]) or "none"
    summary = (f"HANDOFF to human agent: {reason}. Request: '{st['user_message']}'. "
               f"{st['step']} step(s) completed (tools: {tools}).")
    return _result(st, status, summary)


def _result(st: dict, status: str, final_answer: str) -> dict:
    refund = 0.0
    if status in ("REFUND", "ESCALATE"):
        for s in reversed(st["steps"]):
            if s["tool"] in ("issue_refund", "escalate_to_human"):
                refund = s["observation"]["refund_inr"]; break
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final_answer,
            "refund_inr": refund, "steps": st["steps"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"]}


def _loop(st: dict, crash_after_tools: int | None = None) -> dict:
    started = time.monotonic()
    while True:
        if time.monotonic() - started >= st["time_limit_seconds"]:               # (c) wall-clock limit
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {st['time_limit_seconds']}s reached")
        if st["llm_calls"] >= st["max_iterations"]:                              # (a) iteration limit
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"no final answer after {st['llm_calls']} model calls")
        resp = mock_llm.chat(st["messages"], TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:                                # (b) token limit: do not run the tool
            return _fallback(st, "FALLBACK_TOKEN_LIMIT",
                             f"token limit {st['token_limit']} exceeded ({st['tokens_used']} used)")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            return _result(st, answer.split(":")[0].strip(), answer)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        fn = TOOL_MAP.get(tc.name)
        try:
            observation = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
        except Exception as exc:                                                   # tool errors become observations
            observation = {"error": f"{type(exc).__name__}: {exc}"}
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                               "content": json.dumps(observation)})
        st["step"] += 1
        st["tools_executed"].append(tc.name)
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": observation})
        save_checkpoint(st)
        if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")


def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6, token_limit: int = 6000,
              time_limit_seconds: float = 10, run_id: str | None = None, crash_after_tools: int | None = None) -> dict:
    st = {"run_id": run_id or f"{scenario_id}-run", "scenario_id": scenario_id, "user_message": user_message,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "step": 0, "llm_calls": 0, "tools_executed": [], "steps": [],
          "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(st, crash_after_tools)


def resume_agent(run_id: str) -> dict:
    with open(_ckpt_path(run_id), encoding="utf-8") as f:
        st = json.load(f)
    return _loop(st)


# --------------------------------------------------------------------------- driver
def _jsonl(path: str, result: dict) -> None:
    keep = {k: result[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(keep) + "\n")


def main() -> None:
    for p in (RUNS, DEMOS, LEDGER):
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(STATE_DIR):
        for f in os.listdir(STATE_DIR):
            os.remove(os.path.join(STATE_DIR, f))
    os.makedirs("outputs", exist_ok=True)

    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    print("scenario | status | refund INR | model calls | tokens")
    messages = {}
    for r in sc.itertuples():
        messages[r.scenario_id] = r.message
        mock_llm.reset()
        res = run_agent(r.message, r.scenario_id)
        _jsonl(RUNS, res)
        print(f"{r.scenario_id} | {res['status']} | {res['refund_inr']:.2f} | {res['llm_calls']} | {res['tokens_used']}")
        expected = r.expected_outcome
        assert res["status"].startswith(expected), (r.scenario_id, res["status"], expected)
        assert abs(res["refund_inr"] - float(r.expected_refund_inr)) <= 0.01 or expected == "FALLBACK", r.scenario_id

    s1 = messages["S1"]
    d1 = run_agent(s1, "S1", token_limit=200, run_id="S1-tokenlimit"); _jsonl(DEMOS, d1)
    d2 = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-timeout"); _jsonl(DEMOS, d2)
    print(f"\nDemo S1 token_limit=200       -> {d1['status']} (tokens {d1['tokens_used']}, steps {len(d1['steps'])})")
    print(f"Demo S1 time_limit_seconds=0  -> {d2['status']} (llm_calls {d2['llm_calls']})")
    print("   handoff:", d1["final_answer"])

    # crash after the 2nd tool execution, then resume
    mock_llm.reset()
    try:
        run_agent(s1, "S1", run_id="S1-crash", crash_after_tools=2)
        raise AssertionError("crash was not raised")
    except SimulatedCrash as exc:
        print(f"\n{exc}")
    ck = json.load(open(_ckpt_path("S1-crash")))
    print("checkpoint tools_executed:", ck["tools_executed"])
    res = resume_agent("S1-crash"); _jsonl(DEMOS, res)
    print("after resume tools_executed:", [s["tool"] for s in res["steps"]])
    print(f"resume final: {res['status']} {res['refund_inr']:.2f}")
    assert [s["tool"] for s in res["steps"]] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) <= 0.01


if __name__ == "__main__":
    main()
