"""Part A -- audit the raw exports, clean them with explicit rules, and store them (CSV + SQLite).

Run:  python part_a_clean.py
Reads  data/raw/{customers,products,orders}.csv
Writes data/clean/*.csv and data/instamart.db (tables: customers, products, orders)
"""
import os
import sqlite3

import numpy as np
import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
TS_FMT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- audit
def audit_table(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name}: shape={df.shape} ===")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print(summary.to_string())


def audit_raw(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> dict:
    for name, df in (("customers", customers), ("products", products), ("orders", orders)):
        audit_table(name, df)

    amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    age_num = pd.to_numeric(customers["age"], errors="coerce")
    gender_nonblank = customers["gender"].dropna()
    figures = {
        "orders raw rows": len(orders),
        "orders exact duplicate rows": int(orders.duplicated().sum()),
        "orders status == pending": int((orders["status"] == "pending").sum()),
        "orders distinct order_id": int(orders["order_id"].nunique()),
        "order_amount not convertible to number": int((amount_num.isna() & orders["order_amount"].notna()).sum()),
        "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
        "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
        "customers blank age": int(customers["age"].isna().sum()),
        "customers age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
        "customers distinct raw city spellings": int(customers["city"].nunique()),
        "customers distinct non-blank gender spellings": int(gender_nonblank.nunique()),
        "customers blank gender": int(customers["gender"].isna().sum()),
    }
    print("\n=== Audit figures ===")
    for k, v in figures.items():
        print(f"{k:50s} {v:>8,}")
    return figures


# --------------------------------------------------------------------------- customers
GENDER_MAP = {"m": "male", "male": "male", "f": "female", "female": "female"}


def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # (a) city: strip + title case -> exactly six cities
    df["city"] = df["city"].str.strip().str.title()
    # (b) gender: map every spelling to male / female / unknown
    df["gender"] = df["gender"].fillna("").str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    # (c) ages outside 10-100 are invalid -> missing; fill with the city median (groupby + transform)
    n_out = int(((df["age"] < 10) | (df["age"] > 100)).sum())
    df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = np.nan
    n_missing = int(df["age"].isna().sum())
    df["age"] = df["age"].fillna(df.groupby("city")["age"].transform("median")).round().astype(int)
    # (d) signup_date -> datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"], format="%Y-%m-%d")
    print(f"\n[customers] ages set to missing (out of range): {n_out}; ages imputed (total missing): {n_missing}")
    return df


# --------------------------------------------------------------------------- orders
def clean_orders(raw: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    log = {}

    # Rule 1: order_ts -> datetime
    df["order_ts"] = pd.to_datetime(df["order_ts"], format=TS_FMT)

    # Rule 2: remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    log["rule 2: exact duplicate rows removed"] = before - len(df)

    # Rule 3: one record per order_id -- keep the latest order_ts
    before = len(df)
    df = df.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    log["rule 3: superseded (earlier) records removed"] = before - len(df)

    # Rule 4: order_amount -> numeric, then fill from product price
    amount = pd.to_numeric(df["order_amount"], errors="coerce")
    n_bad = int(amount.isna().sum())
    price = df["product_id"].map(products.set_index("product_id")["price_inr"])
    computed = (price * df["quantity"] * (1 - df["discount_pct"] / 100)).round(2)
    df["order_amount"] = amount.fillna(computed).round(2)
    log["rule 4: unparseable order_amount refilled from price"] = n_bad

    # Rule 5: negative delivery -> missing; impute completed orders by category median; leave cancelled missing
    neg = df["delivery_minutes"] < 0
    log["rule 5a: negative delivery_minutes set to missing"] = int(neg.sum())
    df.loc[neg, "delivery_minutes"] = np.nan
    df["category"] = df["product_id"].map(products.set_index("product_id")["category"])
    completed = df["status"] == "completed"
    log["rule 5b: completed orders with missing delivery_minutes (imputed)"] = int((completed & df["delivery_minutes"].isna()).sum())
    sub = df.loc[completed].copy()
    sub["delivery_minutes"] = sub["delivery_minutes"].fillna(sub.groupby("category")["delivery_minutes"].transform("median"))
    df.loc[completed, "delivery_minutes"] = sub["delivery_minutes"]
    df = df.drop(columns="category").sort_values("order_id").reset_index(drop=True)

    print("\n=== Order cleaning log ===")
    for k, v in log.items():
        print(f"{k:70s} {v:>6,}")
    return df


# --------------------------------------------------------------------------- main
def main() -> None:
    os.makedirs(CLEAN, exist_ok=True)
    # Only truly blank cells are missing; strings such as "N/A" must survive so the audit can count them.
    na = dict(keep_default_na=False, na_values=[""])
    customers_raw = pd.read_csv(f"{RAW}/customers.csv", **na)
    products = pd.read_csv(f"{RAW}/products.csv", **na)
    orders_raw = pd.read_csv(f"{RAW}/orders.csv", **na)

    audit_raw(customers_raw, products, orders_raw)
    customers = clean_customers(customers_raw)
    orders = clean_orders(orders_raw, products)

    # Persist: CSV (timestamps as text) and SQLite
    customers_out = customers.assign(signup_date=customers["signup_date"].dt.strftime("%Y-%m-%d"))
    orders_out = orders.assign(order_ts=orders["order_ts"].dt.strftime(TS_FMT))
    customers_out.to_csv(f"{CLEAN}/customers.csv", index=False)
    products.to_csv(f"{CLEAN}/products.csv", index=False)
    orders_out.to_csv(f"{CLEAN}/orders.csv", index=False)

    if os.path.exists(DB):
        os.remove(DB)
    with sqlite3.connect(DB) as con:
        customers_out.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        orders_out.to_sql("orders", con, index=False)
    print(f"\nWrote {CLEAN}/*.csv and {DB}")
    print("cities:", customers["city"].value_counts().to_dict())
    print("genders:", customers["gender"].value_counts().to_dict())
    print("status:", orders["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
