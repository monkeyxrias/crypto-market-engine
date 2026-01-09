import os
import json
import time
from datetime import datetime
from typing import Dict, Any

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import data
import features
from model import ensure_model_exists, load_model, create_labels

from alerts import send_discord_alert


# ==============================
# PERSISTENT COOLDOWN STATE (NOT SESSION-BASED)
# ==============================
ALERT_STATE_FILE = ".alert_state.json"


def _load_alert_state() -> Dict[str, Any]:
    try:
        if os.path.exists(ALERT_STATE_FILE):
            with open(ALERT_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_alert_state(state: Dict[str, Any]) -> None:
    try:
        with open(ALERT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        # Streamlit Cloud FS can be ephemeral; don't crash if write fails
        pass


def _now_ts() -> int:
    return int(time.time())


def _should_send_with_cooldown(state: Dict[str, Any], key: str, cooldown_s: int) -> bool:
    last_ts = int(state.get(key, 0) or 0)
    return (_now_ts() - last_ts) >= cooldown_s


def _mark_sent(state: Dict[str, Any], key: str) -> None:
    state[key] = _now_ts()


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

# --- Discord UI ---
st.sidebar.markdown("---")
st.sidebar.subheader("Discord Alerts")

enable_alerts = st.sidebar.toggle("Discord alerts", value=False)

cooldown_minutes = st.sidebar.slider("Alert cooldown (minutes)", 1, 120, 15)
cooldown_s = int(cooldown_minutes) * 60

# Persistent test output that survives reruns/refresh
if "discord_test_status" not in st.session_state:
    st.session_state.discord_test_status = None  # ("pending"/"ok"/"fail", message)

if st.sidebar.button("🔥 Send test Discord alert", use_container_width=True, key="discord_test_btn"):
    st.session_state.discord_test_status = ("pending", "Sending test alert...")

    webhook = st.secrets.get("DISCORD_WEBHOOK_URL", "")
    if not webhook:
        st.session_state.discord_test_status = ("fail", "DISCORD_WEBHOOK_URL secret is missing/empty.")
    else:
        ok, msg = send_discord_alert(
            webhook_url=webhook,
            content="🔥 TEST: Discord alerts are working! (Crypto Market Engine / Streamlit Cloud)"
        )
        st.session_state.discord_test_status = ("ok", msg) if ok else ("fail", msg)

# Show test status (always visible)
status = st.session_state.discord_test_status
if status:
    kind, msg = status
    if kind == "pending":
        st.sidebar.info(msg)
    elif kind == "ok":
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)
        st.sidebar.caption("Tip: HTTP 401/403=invalid webhook, 404=wrong URL, timeout=requests/network")

# Quick secret visibility indicator
st.sidebar.caption(
    "Webhook loaded: " + ("YES ✅" if st.secrets.get("DISCORD_WEBHOOK_URL", "") else "NO ❌")
)

# Auto refresh
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")


# ==============================
# MODE THRESHOLDS
# ==============================
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
ensure_model_exists("BTC-USD")
clf, scaler = load_model()


# ==============================
# SIGNAL HISTORY (SESSION OK FOR CHARTS)
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
# ALERT LOGIC (REFINED)
# - No first-run spam
# - Alerts only on:
#     * ENTRY (BLOCKED -> ALLOWED)
#     * EXIT  (ALLOWED -> BLOCKED)
#     * EXPOSURE CHANGE (0 <-> 1)
# - Cooldown still applies
# - Persistent across reruns
# ==============================
alert_state = _load_alert_state()

alert_key_state = f"state::{ticker}::{mode}"
alert_key_sent = f"last_sent::{ticker}::{mode}"

current_state = {
    "allow": bool(allow),
    "ma_long": int(ma_long),
    "exposure": float(exposure),
    "p_trend": float(p_trend),
    "p_unknown": float(p_unknown),
}

previous_state = alert_state.get(alert_key_state)

if previous_state is None:
    # First run for this ticker/mode: store but do NOT alert
    alert_state[alert_key_state] = current_state
    _save_alert_state(alert_state)
else:
    prev_allow = bool(previous_state.get("allow", False))
    prev_exposure = float(previous_state.get("exposure", 0.0))

    allow_flip = (prev_allow != bool(allow))
    exposure_flip = (prev_exposure != float(exposure))

    entry_event = (prev_allow is False and bool(allow) is True)
    exit_event = (prev_allow is True and bool(allow) is False)

    # Optional: set True if you only want "ENTRY" alerts
    alert_on_entries_only = False

    if alert_on_entries_only:
        important_change = entry_event or (prev_exposure == 0.0 and float(exposure) > 0.0)
    else:
        important_change = allow_flip or exposure_flip

    if enable_alerts and important_change and _should_send_with_cooldown(alert_state, alert_key_sent, cooldown_s):
        if entry_event:
            event_label = "ENTRY ✅ (Gate opened)"
        elif exit_event:
            event_label = "EXIT ⛔ (Gate closed)"
        elif exposure_flip:
            event_label = "EXPOSURE CHANGE 🔁"
        else:
            event_label = "UPDATE"

        msg = (
            f"**Market Engine Alert** — {event_label}\n"
            f"- Asset/Mode: {ticker} ({mode})\n"
            f"- Time: {now_utc}\n"
            f"- Status: {'ALLOWED ✅' if allow else 'BLOCKED ⛔'}\n"
            f"- Exposure: {exposure:.2f}\n"
            f"- MA Signal: {ma_long:.0f}\n"
            f"- p(Trend): {p_trend:.3f}\n"
            f"- p(Unknown): {p_unknown:.3f}"
        )

        webhook = st.secrets.get("DISCORD_WEBHOOK_URL", "")
        ok, info = send_discord_alert(webhook_url=webhook, content=msg)

        if ok:
            _mark_sent(alert_state, alert_key_sent)
        else:
            st.sidebar.error(f"Alert failed: {info}")

    # Always store current state
    alert_state[alert_key_state] = current_state
    _save_alert_state(alert_state)


# ==============================
# RECORD SIGNAL (FOR CHARTS)
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
# MARKET CONTEXT
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
st.caption("Note: **Unknown** is normal and usually means no strong trend or extreme volatility is detected.")


# ==============================
# RECENT CONFIDENCE
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
