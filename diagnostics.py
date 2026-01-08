import numpy as np
import pandas as pd

import data
import features
from model import load_model, create_labels

def run_diagnostics(n_buckets=5):
    clf, scaler = load_model()

    # Load & prep data
    df = data.get_btc_data()
    df = features.compute_features(df)
    df = create_labels(df)

    # Predict probabilities for each row
    X = df[["return", "volatility", "trend"]].values
    Xs = scaler.transform(X)

    probs = clf.predict_proba(Xs)
    classes = clf.classes_

    # Put probs into dataframe columns
    for i, c in enumerate(classes):
        df[f"p_{c}"] = probs[:, i]

    # We need FUTURE return to judge edge (next 12 hours)
    horizon = 12
    df["fwd_return"] = df["Close"].pct_change(horizon).shift(-horizon)
    df = df.dropna()

    # Focus on Trend probability as a signal
    if "p_Trend" not in df.columns:
        print("No p_Trend column found. Classes were:", classes)
        return

    # Bucket by trend probability
    df["trend_bucket"] = pd.qcut(df["p_Trend"], q=n_buckets, duplicates="drop")

    summary = df.groupby("trend_bucket").agg(
        count=("fwd_return", "count"),
        avg_fwd_return=("fwd_return", "mean"),
        med_fwd_return=("fwd_return", "median"),
        avg_trend_prob=("p_Trend", "mean"),
    ).reset_index()

    print("\n=== Trend-probability buckets vs future return ===")
    print(summary.to_string(index=False))

    # Also show: how often model says Unknown is dominant
    if "p_Unknown" in df.columns:
        dominant = df[[f"p_{c}" for c in classes]].idxmax(axis=1)
        unknown_share = (dominant == "p_Unknown").mean()
        print(f"\nShare of time UNKNOWN is top prediction: {unknown_share:.2%}")

if __name__ == "__main__":
    run_diagnostics()
