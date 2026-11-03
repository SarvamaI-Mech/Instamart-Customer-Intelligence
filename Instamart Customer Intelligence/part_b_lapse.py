"""Part B (tasks 4-8) -- lapse prediction without leakage.

Features use ONLY completed/cancelled orders dated before the 2025-09-01 cutoff plus the clean customer table.
Segment labels from part_b_segments.py are NOT features. Every encoder / scaler / resampler sees training rows only.

Run:  python part_b_lapse.py        (needs outputs/customer_segments.csv)
Writes outputs/lapse_model.joblib and outputs/retention_list.csv
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")
SEED = 42
NUMERIC = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
NOMINAL = ["city", "gender", "fav_category"]


# --------------------------------------------------------------------------- data
def build_dataset() -> pd.DataFrame:
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("data/clean/products.csv")
    orders = orders.merge(products[["product_id", "category"]], on="product_id")

    # label uses September; features use strictly earlier orders
    sept = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END) & (orders["status"] == "completed")]
    before = orders[orders["order_ts"] < CUTOFF]
    done = before[before["status"] == "completed"]

    feat = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"), frequency=("order_id", "nunique"), monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"), avg_delivery=("delivery_minutes", "mean"))
    feat["recency"] = (CUTOFF - feat["last_order"]).dt.days
    n_final = before[before["status"].isin(["completed", "cancelled"])].groupby("customer_id")["order_id"].nunique()
    n_cancel = before[before["status"] == "cancelled"].groupby("customer_id")["order_id"].nunique()
    feat["cancel_rate"] = (n_cancel.reindex(feat.index).fillna(0) / n_final.reindex(feat.index)).astype(float)
    # favourite category: most frequent; ties broken alphabetically for determinism
    fav = (done.groupby(["customer_id", "category"]).size().reset_index(name="n")
           .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
           .drop_duplicates("customer_id").set_index("customer_id")["category"])
    feat["fav_category"] = fav

    feat = feat.join(customers.set_index("customer_id")[["city", "gender", "age", "signup_date"]])
    feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
    feat["lapsed"] = (~feat.index.isin(sept["customer_id"])).astype(int)
    return feat[NUMERIC + NOMINAL + ["lapsed"]]


def prep() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL),
    ])


def lapsed_metrics(y_true, y_pred) -> dict:
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def fmt(m: dict) -> str:
    return f"precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}"


# --------------------------------------------------------------------------- main
def main() -> None:
    data = build_dataset()
    X, y = data[NUMERIC + NOMINAL], data["lapsed"]
    assert X.isna().sum().sum() == 0, "feature columns contain missing values"
    print(f"cohort={len(data):,} lapsed={y.sum()} ({y.mean():.2%}); no missing feature values: True")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train={len(X_tr)} test={len(X_te)}")

    # (a) baseline
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    bp = base.predict(X_te)
    print(f"\nBaseline 'never lapsed': accuracy={accuracy_score(y_te, bp):.3%} lapsed-F1={f1_score(y_te, bp, zero_division=0):.3f}")

    # (b) logistic regression
    lr_gs = GridSearchCV(Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))]),
                         {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
    # (c) random forest
    rf_gs = GridSearchCV(Pipeline([("prep", prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=SEED))]),
                         {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(X_tr, y_tr)
    print(f"\nLogReg best {lr_gs.best_params_} CV-F1={lr_gs.best_score_:.3f}")
    print(f"RandForest best {rf_gs.best_params_} CV-F1={rf_gs.best_score_:.3f}")

    # preprocessing leakage check
    scaler = lr_gs.best_estimator_.named_steps["prep"].named_transformers_["num"]
    fi = NUMERIC.index("frequency")
    print(f"\nScaler learned mean of frequency (train only)={scaler.mean_[fi]:.4f}  full-cohort mean={X['frequency'].mean():.4f}")

    # (d) test evaluation
    results = {}
    for name, gs in (("LogReg", lr_gs), ("RandForest", rf_gs)):
        pred = gs.predict(X_te)
        results[name] = lapsed_metrics(y_te, pred)
        print(f"\n{name} test: {fmt(results[name])}")
        print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, pred))

    # model choice is made on cross-validated F1 (not on the test set)
    sel_name, sel_gs = max((("LogReg", lr_gs), ("RandForest", rf_gs)), key=lambda t: t[1].best_score_)
    selected = sel_gs.best_estimator_
    assert results[sel_name]["f1"] >= 0.55, "selected model below the required test F1"
    print(f"\nSelected model: {sel_name} (highest CV F1); test F1={results[sel_name]['f1']:.3f}")

    # class-balance experiment: oversample lapsed rows of the TRAINING set only
    rng = np.random.RandomState(SEED)
    pos = np.flatnonzero(y_tr.to_numpy() == 1)
    neg = np.flatnonzero(y_tr.to_numpy() == 0)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    idx = np.concatenate([np.arange(len(y_tr)), extra])
    X_bal, y_bal = X_tr.iloc[idx], y_tr.iloc[idx]
    print(f"\nBalanced training counts: {y_bal.value_counts().to_dict()}")
    lr_bal = clone(lr_gs.best_estimator_).fit(X_bal, y_bal)
    m_plain, m_bal = results["LogReg"], lapsed_metrics(y_te, lr_bal.predict(X_te))
    print("LogReg un-balanced:", fmt(m_plain))
    print("LogReg balanced   :", fmt(m_bal))
    print("change:", {k: round(m_bal[k] - m_plain[k], 3) for k in m_plain})

    # interpretation
    rf_pipe = rf_gs.best_estimator_
    names = rf_pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf_pipe.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    print("\nTop-5 random-forest importances:\n", imp.head(5).round(4).to_string())
    coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0],
                     index=lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out())
    for f in ("num__recency", "num__frequency", "num__monetary"):
        print(f"LogReg coefficient {f}: {coef[f]:+.3f}")

    # persistence
    joblib.dump(selected, "outputs/lapse_model.joblib")
    reloaded = joblib.load("outputs/lapse_model.joblib")
    same = np.array_equal(reloaded.predict(X_te), selected.predict(X_te)) and \
        np.allclose(reloaded.predict_proba(X_te), selected.predict_proba(X_te))
    assert same
    print("\nReloaded model reproduces test predictions exactly:", same)

    # retention list (illustrative: Premium flag is a post-hoc whole-window label)
    seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
    proba = pd.Series(reloaded.predict_proba(X_te)[:, 1], index=X_te.index, name="lapse_probability")
    lst = pd.DataFrame({"lapse_probability": proba, "actually_lapsed": y_te})
    lst = lst[lst.index.map(lambda c: bool(seg["is_premium"].get(c, False)))]
    lst = lst.sort_values("lapse_probability", ascending=False).head(10)
    lst.round(4).rename_axis("customer_id").reset_index().to_csv("outputs/retention_list.csv", index=False)
    print(f"Retention list: {len(lst)} Premium test customers, {int(lst['actually_lapsed'].sum())} truly lapsed")


if __name__ == "__main__":
    main()
