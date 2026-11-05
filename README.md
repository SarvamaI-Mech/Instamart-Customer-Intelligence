# Instamart Customer Intelligence & Support Copilot

An end-to-end data, ML, and agentic support system for quick-commerce operations. It addresses three problems: untrusted order exports, customer churn blindness, and unstructured support tickets. All data is synthetic (INR, `seed=42`) and runs offline via `mock_llm.py`, with no external API costs.

## Modules

| Module | Deliverable | Key Results |
| :-- | :-- | :-- |
| **A. Data Foundation** | Audit and clean raw exports into SQLite; 7 SQL queries and 14 integrity checks | 13,084 → 12,309 distinct orders; INR 2,998,668.80 validated across 11,110 completed orders |
| **B. Customer ML** | RFM + K-Means (K=4); leakage-free Random Forest lapse model | Silhouette 0.526; Test F1 0.655 (beats Logistic Regression and baseline) |
| **C. Ticket Triage** | PII redaction, retries, token budgets, routing | 278 PII instances redacted; 147/147 accuracy; 19/19 transient errors recovered |
| **D. RAG & Agent** | TF-IDF policy assistant; resumable ReAct refund agent | 12/12 policy eval accuracy; zero duplicate tool calls after crash/resume |

## Tech Stack

Python 3.11+, pandas, numpy, sqlite3, scikit-learn, joblib, offline `mock_llm.py`

## Project Structure

```text
instamart/
├── data/                 # Raw/clean CSVs and instamart.db
├── sql/queries.sql       # 7 analytical queries
├── outputs/              # Segments, model, triage results, ledgers
├── state/                # Agent checkpoints
├── part_a_*.py           # Cleaning, SQL, EDA
├── part_b_*.py           # Segmentation, lapse model
├── part_c_triage.py      # Ticket triage
├── part_d_*.py           # RAG and agent
├── validate_checkpoints.py
├── run_all.sh
└── requirements.txt
```

## Getting Started

```bash
git clone <repo-url> instamart && cd instamart
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run_all.sh
```

Each stage can also be run on its own, starting with `python generate_data.py`, then `part_a_clean.py`, and so on.

## Engineering Highlights

- **Data hygiene:** Removed 119 duplicates and 656 superseded `pending` logs, and recomputed totals for 120 corrupted amounts.
- **No leakage:** Strict `2025-09-01` temporal cutoff, with encoding and scaling fitted inside a `Pipeline` on training folds only.
- **Guardrails:** Regex PII redaction, output validation, and token and timeout circuit breakers.
- **Resilient agent:** 5 tools, human escalation for refunds above INR 1,000, and JSON checkpoints after every tool call.

## Limitations & Next Steps

- RFM `is_premium` tags and category delivery medians include September data and should be computed pre-cutoff in production.
- TF-IDF misses semantic synonyms (e.g., "money back" vs. "refund"), so dense embeddings are the next step.
- Further work: evaluation against a real LLM and idempotency keys for the refund ledger.

## Author

**Saravanan M**
