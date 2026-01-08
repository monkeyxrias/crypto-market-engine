import numpy as np
import pandas as pd

import data
import features
from model import load_model, create_labels


def compute_ma_signals(df, fast=24, slow=72):
    df = df.copy()
    df["ma_fast"] = df["Close"].rolling(fast).mean()
    df["ma_slow"] = df["Close"].rolling(slow).mean()
    df["ma_long"] = (df["ma_fast"] > df["ma_slow"]).astype(float)
    return df


def max_drawdown(equity_series: pd.Series) -> float:
    peak = equity_series.cummax()
    dd = equity_series / peak - 1.0
    return float(dd.min())  # negative number, e.g. -0.25


def run_once(
    df_base: pd.DataFrame,
    probs: np.ndarray,
    classes: np.ndarray,
    trend_on_threshold: float,
    unknown_off_threshold: float,
):
    # returns
    ret = df_base["Close"].pct_change().fillna(0).values
    ma_long = df_base["ma_long"].values.astype(float)

    # map class -> column index
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_trend = class_to_idx.get("Trend", None)
    idx_unknown = class_to_idx.get("Unknown", None)

    eq = [1.0]
    exposure = [0.0]

    for i in range(1, len(df_base)):
        p_trend = probs[i, idx_trend] if idx_trend is not None else 0.0
        p_unknown = probs[i, idx_unknown] if idx_unknown is not None else 0.0

        allow = (p_trend >= trend_on_threshold) and (p_unknown < unknown_off_threshold)
        exp = ma_long[i] if allow else 0.0

        exposure.append(exp)
        eq.append(eq[-1] * (1 + exp * ret[i]))

    eq_s = pd.Series(eq, index=df_base.index[: len(eq)])
    exposure_s = pd.Series(exposure, index=df_base.index[: len(exposure)])

    final_eq = float(eq_s.iloc[-1])
    mdd = max_drawdown(eq_s)
    time_in_mkt = float((exposure_s > 0).mean())

    return final_eq, mdd, time_in_mkt


def main():
    # --- Load model ---
    clf, scaler = load_model()

    # --- Build dataset once ---
    df = data.get_btc_data()
    df = features.compute_features(df)
    df = create_labels(df)
    df = compute_ma_signals(df, fast=24, slow=72)

    # drop early NaNs from MAs
    df = df.dropna().copy()

    # --- Precompute probabilities for every row once (fast) ---
    X = df[["return", "volatility", "trend"]].values
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)
    classes = clf.classes_

    # --- Baselines (for reference) ---
    df["ret"] = df["Close"].pct_change().fillna(0)

    # Buy & Hold equity
    eq_bh = (1 + df["ret"]).cumprod()
    bh_final = float(eq_bh.iloc[-1])
    bh_mdd = max_drawdown(eq_bh)

    # MA-only equity
    eq_ma = (1 + df["ma_long"] * df["ret"]).cumprod()
    ma_final = float(eq_ma.iloc[-1])
    ma_mdd = max_drawdown(eq_ma)

    print("\n=== Baselines ===")
    print(f"Buy & Hold: final={bh_final:.4f}, maxDD={bh_mdd:.2%}")
    print(f"MA only  : final={ma_final:.4f}, maxDD={ma_mdd:.2%}")

    # --- Grid search thresholds ---
    trend_grid = np.round(np.arange(0.35, 0.81, 0.05), 2)      # 0.35..0.80
    unknown_grid = np.round(np.arange(0.30, 0.76, 0.05), 2)    # 0.30..0.75

    results = []

    for t_thr in trend_grid:
        for u_thr in unknown_grid:
            final_eq, mdd, time_in_mkt = run_once(
                df_base=df,
                probs=probs,
                classes=classes,
                trend_on_threshold=t_thr,
                unknown_off_threshold=u_thr,
            )

            # Simple score: prefer higher final equity and lower drawdown
            # (You can adjust this later; this is a good starting point)
            score = final_eq + (mdd * 0.5)  # mdd is negative, so this penalizes drawdown

            results.append({
                "trend_thr": t_thr,
                "unknown_thr": u_thr,
                "final_equity": final_eq,
                "max_drawdown": mdd,
                "time_in_mkt": time_in_mkt,
                "score": score
            })

    res = pd.DataFrame(results)

    # Sort by score (best first)
    res = res.sort_values("score", ascending=False)

    print("\n=== Top 15 threshold settings (ranked) ===")
    print(
        res.head(15)[
            ["trend_thr", "unknown_thr", "final_equity", "max_drawdown", "time_in_mkt", "score"]
        ].to_string(index=False)
    )

    # Also show top by raw final equity (ignoring drawdown)
    res2 = res.sort_values("final_equity", ascending=False)
    print("\n=== Top 10 by FINAL EQUITY only ===")
    print(
        res2.head(10)[
            ["trend_thr", "unknown_thr", "final_equity", "max_drawdown", "time_in_mkt"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
