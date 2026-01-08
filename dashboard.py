import streamlit as st
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

import data
import features
from model import ensure_model_exists, load_model, create_labels

# ==============================
# PAGE SETUP
# ==============================
st.set_page_config(page_title="Crypto Market Engine", layout="wide")
st.title("Crypto Market Engine")
st.caption("Decision support tool • MA baseline + AI regime filter (not auto-trading)")

# ==============================
# SIDEBAR SETTINGS
# ==============================
st.sidebar.header("Settings")

ticker = st.sidebar.selectbox(
    "Asset",
    ["BTC-USD", "ETH-USD", "SOL-USD"],
    index=0
)

mode = st.sidebar.selectbox(
    "Risk profile",
    ["balanced", "conservative", "ma_only"],
    index=0
)

refresh_minutes = st.sidebar.slider(
    "Auto-refresh (minutes)", min_value=1, max_value=30, value=5
)

# Auto refresh
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

# Mode thresholds
if mode == "conservative":
    trend_thr = 0.75
    unknown_thr = 0.50
elif mode == "balanced":
    trend_thr = 0.50
    unknown_thr = 0.50
else:
    trend_thr = None
    unknown_thr = None

# ==============================
# ENSURE MODEL EXISTS (DEPLOY SAFE)
# ==============================
# Train+save model on first cloud run if missing
ensure_model_exists("BTC-USD")

# Load model (no caching until deployment is stable)
clf, scaler = load_model()

# ==============================
# SIGNAL HISTORY
# ==============================
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

# ==============================
# LOAD DATA
# ==============================
df = data.get_price_data(ticker)
df = features.compute_features(df)
df = create_labels(df)

# ==============================
# MOVING AVERAGE SIGNAL
# ==============================
fast = 24
slow = 72

df_ma = df.copy()
df_ma["ma_fast"] = df_ma["Close"].rolling(fast).mean()
df_ma["ma_slow"] = df_ma["Close"].rolling(slow).mean()
df_ma = df_ma.dropna()

ma_long = float(df_ma["ma_fast"].iloc[-1] > df_ma["ma_slow"].iloc[-1])

# ==============================
# AI REGIME PROBABILITIES
# ==============================
latest = df.iloc[-1]

X_latest = np.array(
    [float(latest["return"]), float(latest["volatility"]), float(latest["trend"])]
).reshape(1, -1)

X_scaled = scaler.transform(X_latest)

probs = clf.predict_proba(X_scaled)[0]
classes = clf.classes_
prob = dict(zip(classes, probs))

p_trend = float(prob.get("Trend", 0.0))
p_unknown = float(prob.get("Unknown", 0.0))

# ==============================
# DECISION LOGIC
# ==============================
if mode == "ma_only":
    allow = True
    exposure = ma_long
else:
    allow = (p_trend >= trend_thr) and (p_unknown < unknown_thr)
    exposure = ma_long if allow else 0.0

now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# ==============================
# RECORD SIGNAL
# ==============================
signal = {
    "time": now_utc,
    "ticker": ticker,
    "mode": mode,
    "ma_signal": int(ma_long),
    "p_trend": round(p_trend, 3),
    "p_unknown": round(p_unknown, 3),
    "allowed": bool(allow),
    "exposure": round(float(exposure), 2),
}
st.session_state.signal_history.append(signal)
st.session_state.signal_history = st.session_state.signal_history[-200:]

# ==============================
# TOP DECISION PANEL
# ==============================
st.subheader("Decision")

col1, col2, col3, col4, col5 = st.columns([1.3, 1, 1, 1, 1.5])

status_text = "ALLOWED ✅" if allow else "BLOCKED ⛔"

col1.metric("Status", status_text)
col2.metric("Exposure", f"{exposure:.2f}")
col3.metric("MA Signal", f"{ma_long:.0f}")
col4.metric("Mode", mode)
col5.metric("Last update", now_utc)

if mode != "ma_only":
    st.caption(f"Gate: p(Trend) ≥ {trend_thr} AND p(Unknown) < {unknown_thr}")

# ==============================
# MARKET CONTEXT (HUMAN EXPLANATION)
# ==============================
if p_trend >= 0.55:
    context = "Trend conditions detected — directional moves are more likely."
elif p_unknown >= 0.50:
    context = "Neutral / consolidation environment — edge is low (this is normal)."
else:
    context = "Transitional or unstable market — conditions are uncertain."

st.info(f"**Market context:** {context}")

# ==============================
# FEATURE SNAPSHOT
# ==============================
st.subheader("Latest Features")

f1, f2, f3 = st.columns(3)
f1.metric("Trend", f"{float(latest['trend']):.5f}")
f2.metric("Volatility", f"{float(latest['volatility']):.5f}")
f3.metric("Return", f"{float(latest['return']):.5f}")

# ==============================
# REGIME CONFIDENCE (NOW)
# ==============================
st.subheader("Market Regime Confidence (Neutral is common)")

probs_df = pd.DataFrame({"Regime": classes, "Probability": probs}).sort_values("Probability", ascending=False)
st.bar_chart(probs_df.set_index("Regime")["Probability"])

st.caption("Note: **Unknown** is a normal state and usually means no strong trend or extreme volatility is detected.")

# ==============================
# REGIME CONFIDENCE (RECENT)
# ==============================
st.subheader("Market Regime Confidence (Recent)")

hist_df = pd.DataFrame(st.session_state.signal_history)

if not hist_df.empty:
    hist_df = hist_df[(hist_df["ticker"] == ticker) & (hist_df["mode"] == mode)].copy()
    hist_df["time_dt"] = pd.to_datetime(hist_df["time"], format="%Y-%m-%d %H:%M UTC", errors="coerce")
    hist_df = hist_df.dropna(subset=["time_dt"]).sort_values("time_dt").tail(80)

    line_df = hist_df.set_index("time_dt")[["p_trend", "p_unknown"]]
    st.line_chart(line_df)
else:
    st.info("No signal history yet. Leave the dashboard open for a few refreshes.")

# ==============================
# HOW TO READ THIS
# ==============================
st.markdown("### How to read this")
st.markdown(
    """
- **Trend** — the model detects statistically strong directional movement.
- **High Volatility** — the market is unstable; risk is elevated.
- **Unknown** — neutral conditions (most common), often consolidation.

**Design note:**  
This system is intentionally quiet. When **Trend** confidence rises and **Unknown** falls,
conditions are usually more favorable.
"""
)

# ==============================
# PRICE CHART
# ==============================
st.subheader(f"{ticker} Price (Recent)")

fig, ax = plt.subplots()
df["Close"].tail(250).plot(ax=ax)
ax.set_ylabel("Price")
ax.set_xlabel("Time")
st.pyplot(fig)

# ==============================
# SIGNAL HISTORY TABLE
# ==============================
with st.expander("Recent Signals (details)"):
    if not hist_df.empty:
        st.dataframe(hist_df.drop(columns=["time_dt"], errors="ignore")[::-1].tail(50), use_container_width=True)
    else:
        st.write("No signals yet.")
