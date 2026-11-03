"""Part A -- pandas cross-check of Q2 and exploratory statistics of completed order_amount."""
import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

done = orders[orders["status"] == "completed"]
df = done.merge(customers[["customer_id", "city"]], on="customer_id").merge(
    products[["product_id", "category"]], on="product_id")

# (a) Q2 rebuilt with pivot_table(margins=True)
pivot = pd.pivot_table(df, index="city", columns="category", values="order_amount", aggfunc="sum", margins=True)
print("Pivot grand total INR %.2f (SQL Q2 total: 2,998,668.80)" % pivot.loc["All", "All"])
assert round(pivot.loc["All", "All"], 2) == 2998668.80
print(pivot.round(2).to_string())

by_city = df.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                 count=("order_amount", "count")).round(2)
print("\nPer-city sum / mean / count:\n", by_city.to_string())

# (b) distribution statistics
x = done["order_amount"]
q1, q2_, q3 = x.quantile([.25, .5, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean={x.mean():.2f} median={x.median():.2f} skew={x.skew():.2f}")
print(f"Q1={q1:.2f} Q2={q2_:.2f} Q3={q3:.2f} IQR={iqr:.2f}")
print(f"fences lower={lo:.2f} upper={hi:.2f}; below={int((x < lo).sum())} above={int((x > hi).sum())}")

corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\nCorrelation matrix:\n", corr.round(3).to_string())
pairs = corr.where(~(corr.index.to_numpy()[:, None] == corr.columns.to_numpy()[None, :])).stack()
best = pairs.abs().idxmax()
print(f"Largest |correlation|: {best[0]} vs {best[1]} = {corr.loc[best]:.3f}")
