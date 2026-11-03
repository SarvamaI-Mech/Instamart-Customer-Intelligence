"""Part A -- assert the reference checkpoints of the clean data. Prints ALL CHECKPOINTS PASSED on success."""
import sqlite3

import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

checks = {
    "12,309 order rows (one per order_id)": len(orders) == 12309 and orders["order_id"].is_unique,
    "11,110 completed orders": len(completed) == 11110,
    "1,199 cancelled orders": len(cancelled) == 1199,
    "no pending rows": (orders["status"] == "pending").sum() == 0,
    "0 missing order_amount": orders["order_amount"].isna().sum() == 0,
    "0 missing delivery_minutes among completed": completed["delivery_minutes"].isna().sum() == 0,
    "1,199 missing delivery_minutes among cancelled": cancelled["delivery_minutes"].isna().sum() == 1199,
    "completed revenue = 2,998,668.80": round(completed["order_amount"].sum(), 2) == 2998668.80,
    "six cities": customers["city"].nunique() == 6,
    "city counts": customers["city"].value_counts().to_dict()
        == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages, all within 18-60": customers["age"].isna().sum() == 0 and customers["age"].between(18, 60).all(),
}
with sqlite3.connect("data/instamart.db") as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    checks["database has customers/products/orders"] = {"customers", "products", "orders"} <= tables
    checks["db revenue matches CSV"] = round(con.execute(
        "SELECT SUM(order_amount) FROM orders WHERE status='completed'").fetchone()[0], 2) == 2998668.80

failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + name)
assert not failed, f"Failed checkpoints: {failed}"
print("ALL CHECKPOINTS PASSED")
