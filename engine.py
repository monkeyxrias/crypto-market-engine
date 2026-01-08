import numpy as np

import data
import features
from model import load_model, create_labels

def compute_ma_signal(df, fast=24, slow=72):
    df = df.copy()
    df["ma_fast"] = df["Close"].rolling(fast).mean()
    df["ma_slow"] = df["Close"].rolling(slow).mean()
    df = df.dropna()
    ma_long = float(df["ma_fast"].iloc[-1] > df["ma_slow"].iloc[-1])
    return ma_long, df

def run_engine(mode="balanced"):
    clf, scaler = load_model()

    df = data.get_btc_data()
    df = features.compute_features(df)
    df = create_labels(df)

    # MA signal on the same df (needs Close)
    ma_long, df_ma = compute_ma_signal(df, fast=24, slow=72)

    latest = df.iloc[-1]
    X = np.array([[latest["return"], latest["volatility"], latest["trend"]]])
    Xs = scaler.transform(X)

    probs = clf.predict_proba(Xs)[0]
    classes = clf.classes_
    prob = dict(zip(classes, probs))

    p_trend = prob.get("Trend", 0.0)
    p_unknown = prob.get("Unknown", 0.0)

    # Modes from our grid search
    if mode == "conservative":
        trend_thr = 0.75
        unknown_thr = 0.50
        allow = (p_trend >= trend_thr) and (p_unknown < unknown_thr)
        exposure = ma_long if allow else 0.0

    elif mode == "balanced":
        trend_thr = 0.50
        unknown_thr = 0.50
        allow = (p_trend >= trend_thr) and (p_unknown < unknown_thr)
        exposure = ma_long if allow else 0.0

    elif mode == "ma_only":
        allow = True
        exposure = ma_long

    else:
        raise ValueError("mode must be one of: conservative, balanced, ma_only")

    print("\n==============================")
    print("BTC Market Engine")
    print("==============================")
    print(f"Mode: {mode}")
    print(f"MA signal (long=1 / flat=0): {ma_long}")
    print(f"p(Trend):   {p_trend:.3f}")
    print(f"p(Unknown): {p_unknown:.3f}")
    print(f"Trading allowed: {allow}")
    print(f"Final exposure: {exposure}")
    print("==============================\n")

if __name__ == "__main__":
    run_engine(mode="balanced")
