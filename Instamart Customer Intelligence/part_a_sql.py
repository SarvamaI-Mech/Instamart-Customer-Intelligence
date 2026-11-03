"""Part A -- run the seven queries in sql/queries.sql against data/instamart.db and print the stated results."""
import re
import sqlite3

import pandas as pd

DB = "data/instamart.db"


def load_queries(path="sql/queries.sql"):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- (Q\d+):.*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main():
    qs = load_queries()
    with sqlite3.connect(DB) as con:
        r = {k: pd.read_sql_query(v, con) for k, v in qs.items()}

    print(f"Q1 customers with >8000 in orders >500: {len(r['Q1'])}")
    q2 = r["Q2"]
    top = q2.iloc[0]
    print(f"Q2 rows={len(q2)} total={q2['revenue'].sum():,.2f} largest={top['city']}/{top['category']} {top['revenue']:,.2f}")
    print(f"Q3 products above average price: {len(r['Q3'])}")
    print(f"Q4 customers without a completed order: {len(r['Q4'])}")
    print(f"Q5 customers above average spend: {len(r['Q5'])}; average INR {r['Q5']['avg_total'].iloc[0]:,.2f}")
    q6 = r["Q6"]
    print("Q6 top customer per city:")
    print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
    print(f"Q6 rows where RANK() != ROW_NUMBER(): {int((q6['rnk'] != q6['row_num']).sum())}")
    q7 = r["Q7"]
    print(f"Q7 daily rows={len(q7)}; last cumulative INR {q7['cumulative_revenue'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
