"""Part B (tasks 1-3) -- RFM table, K-means model selection, segment profiling and PCA projection.

Run:  python part_b_segments.py
Writes outputs/customer_segments.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
SEED = 42


def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    done = orders[orders["status"] == "completed"]
    rfm = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[["recency", "frequency", "monetary"]].reset_index()


def name_segments(profile: pd.DataFrame) -> dict:
    """Business names derived from each segment's mean RFM (cluster ids carry no order)."""
    names, left = {}, set(profile.index)
    premium = profile["monetary"].idxmax()
    names[premium] = ("Premium Big Spenders", "Concierge perks, early access to electronics launches, 1-hour support SLA")
    left.discard(premium)
    loyal = profile.loc[list(left), "frequency"].idxmax()
    names[loyal] = ("Loyal Regulars", "Basket-building bundles and subscription-style reorder reminders")
    left.discard(loyal)
    lapsing = profile.loc[list(left), "recency"].idxmax()
    names[lapsing] = ("Drifting / At Risk", "Win-back coupon plus a reminder of favourite items before they lapse")
    left.discard(lapsing)
    for c in left:
        names[c] = ("Occasional Shoppers", "Low-cost nudges: free-delivery threshold reminders and category discovery")
    return names


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    rfm = build_rfm(orders)
    print(f"RFM customers: {len(rfm):,}")
    print(f"recency median={rfm['recency'].median():.0f} min={rfm['recency'].min()} max={rfm['recency'].max()}")
    print(f"frequency mean={rfm['frequency'].mean():.2f} max={rfm['frequency'].max()}")
    print(f"monetary mean={rfm['monetary'].mean():.2f} max={rfm['monetary'].max():.2f}")

    X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])

    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
    table = pd.DataFrame(rows)
    print("\nK-means model selection:\n", table.to_string(index=False))

    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=SEED).fit(X)
    rfm["segment_id"] = km.labels_
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
    names = name_segments(profile)
    profile["segment_name"] = [names[i][0] for i in profile.index]
    profile["action"] = [names[i][1] for i in profile.index]
    print(f"\nSegment profile (K={FINAL_K}):\n", profile.drop(columns="action").to_string())
    for i, r in profile.iterrows():
        print(f"  segment {i} -> {r['segment_name']}: {r['action']}")

    rfm["segment_name"] = rfm["segment_id"].map(lambda i: names[i][0])
    premium_id = profile["monetary"].idxmax()
    rfm["is_premium"] = rfm["segment_id"] == premium_id
    assert rfm.groupby("segment_id")["is_premium"].first().sum() == 1

    pca = PCA(n_components=2).fit(X)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: {evr[0]:.3f}, {evr[1]:.3f}, total {evr.sum():.3f}")

    out = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
    out.to_csv("outputs/customer_segments.csv", index=False)
    print(f"Wrote outputs/customer_segments.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
