"""Part D (tasks 1-4) -- chunking, TF-IDF retrieval, grounded answering and evaluation of the policy assistant.

Run:  python part_d_rag.py
The module is also imported by part_d_agent.py (retrieve / answer are reused by the search_policy tool).
"""
import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = "data/policies"
SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for customer support. Answer only from the supplied CONTEXT. "
                 "If the answer is not in the context, say that you could not find the information.")


# --------------------------------------------------------------------------- chunking
def chunk_words(text: str, size: int = 40, overlap: int = 10) -> list:
    """Word windows of `size` words, each starting `size - overlap` words after the previous one.
    Stops after the first chunk whose end reaches or passes the last word."""
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def load_chunks(size: int = 40, overlap: int = 10) -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        for c in chunk_words(text, size, overlap):
            out.append({"source": os.path.basename(path), "text": c})
    return out


# --------------------------------------------------------------------------- retrieval
class Retriever:
    """TF-IDF index over policy chunks; fitted once, queries are transformed with the same vocabulary."""

    def __init__(self, size: int = 40, overlap: int = 10):
        self.chunks = load_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> list:
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]}
                for i in top]


_DEFAULT = None


def retrieve(query: str, k: int = 3) -> list:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever()
    return _DEFAULT.retrieve(query, k)


# --------------------------------------------------------------------------- augment + answer
def answer(question: str, retriever=None):
    hits = (retriever.retrieve(question, 3) if retriever else retrieve(question, 3))
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]


# --------------------------------------------------------------------------- evaluation
def evaluate(size: int, overlap: int, label: str) -> dict:
    r = Retriever(size, overlap)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hit = fact_hit = fallback = 0
    failures = []
    ans_q = ev[ev["expected_doc"] != "NONE"]
    for q in ans_q.itertuples():
        text, sources = answer(q.question, r)
        d, f = q.expected_doc in sources, q.expected_fact in text
        doc_hit += d; fact_hit += f
        if not (d and f):
            failures.append((q.question, d, f, text))
    for q in ev[ev["expected_doc"] == "NONE"].itertuples():
        text, _ = answer(q.question, r)
        fallback += text.startswith("I could not find")
    n_ans, n_un = len(ans_q), len(ev) - len(ans_q)
    print(f"\n[{label}] chunks={len(r.chunks)}  doc-in-top3={doc_hit}/{n_ans}  fact-in-answer={fact_hit}/{n_ans}  fallback={fallback}/{n_un}")
    for q, d, f, t in failures:
        print(f"   FAIL doc_ok={d} fact_ok={f}: {q}\n        answer: {t}")
    return {"chunks": len(r.chunks), "doc": doc_hit, "fact": fact_hit, "fallback": fallback}


def main() -> None:
    r = Retriever()
    per_doc = pd.Series([c["source"] for c in r.chunks]).value_counts().sort_index()
    print(f"chunks total={len(r.chunks)}\n{per_doc.to_string()}")
    hits = r.retrieve("refund policy late delivery")
    print("\nTop-3 for 'refund policy late delivery':", [(h["source"], h["score"]) for h in hits])
    evaluate(40, 10, "overlap 10")
    evaluate(40, 0, "overlap 0")


if __name__ == "__main__":
    main()
