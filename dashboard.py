import os
import json
import time
from datetime import datetime
from typing import Dict, Any

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import altair as alt

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
        pass


def _now_ts() -> int:
    return int(time.time())


def _should_send_with_cooldown(state: Dict[str, Any], key: str, cooldown_s: int) -> bool:
    last_ts = int(state.get(key, 0) or 0)
    return (_now_ts() - last_ts) >= cooldown_s


def _mark_sent(state: Dict[str, Any], key: str) -> None:
    state[key] = _now_ts()


# ==============================
# DATA NORMALIZATION (THE IMPORTANT FIX)
# ==============================
def _extract_close_series(raw: Any) -> pd.Series:
    """
    Return a 1-D Close series from a raw DataFrame, handling:
    - single columns
    - MultiIndex columns (e.g., yfinance)
    - df['Close'] returning a DataFrame
    """
    if raw is None:
        raise ValueError("Price data is None.")
    if not isinstance(raw, pd.DataFrame):
        raise ValueError(f"Price data is not a DataFrame (got {type(raw)}).")
    if raw.empty:
        raise ValueError("Price data is empty.")

    # Simple case
    if "Close" in raw.columns:
        obj = raw["Close"]
        if isinstance(obj, pd.Series):
            return obj
        if isinstance(obj, pd.DataFrame) and obj.shape[1] >= 1:
            return obj.iloc[:, 0]

    # MultiIndex columns (common)
    if isinstance(raw.columns, pd.MultiIndex):
        # Try selecting level-0 name Close
        try:
            close_df = raw.xs("Close", axis=1, level=0, drop_level=False)
            if isinstance(close_df, pd.DataFrame) and close_df.shape[1] >= 1:
                return close_df.iloc[:, 0]
        except Exception:
            pass

        # Fallback: search tuple columns containing "close"
        for col in raw.columns:
            if isinstance(col, tuple) and any(str(x).lower() == "close" for x in col):
                s = raw[col]
                if isinstance(s, pd.Series):
                    return s

    # Fallback: fuzzy search
    for c in raw.columns:
        if "close" in str(c).lower():
            obj = raw[c]
            if isinstance(obj, pd.Series):
                return obj
            if isinstance(obj, pd.DataFrame) and obj.shape[1] >= 1:
                return obj.iloc[:, 0]

    raise ValueError(f"Could not find Close column. Columns sample: {list(raw.columns)[:10]}")


def _normalize_price_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean DF with:
      - index: datetime if possible
      - column: Close (float)
    """
    close = _extract_close_series(raw)
    close = pd.to_numeric(close, errors="coerce")

    # Determine datetime index
    idx = None
    if isinstance(raw.index, pd.DatetimeIndex):
        idx = raw.index
    elif "Date" in raw.columns:
        dt = pd.to_datetime(raw["Date"], errors="coerce")
        if dt.notna().any():
            idx = dt
    else:
        # If close already has datetime-like index, use it
        if isinstance(close.index, pd.DatetimeIndex):
            idx = close.index

    df_clean = pd.DataFrame({"Close": close.values})

    # Apply index
    if idx is not None and len(idx) == len(df_clean):
        df_clean.index = pd.to_datetime(idx, errors="coerce")
        df_clean = df_clean.dropna(axis=0, how="any")  # drop rows where index or close bad
    else:
        # fallback to synthetic time axis
        df_clean = df_clean.dropna(subset=["Close"])
        df_clean.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df_clean), freq="H")

    df_clean = df_clean.sort_index()
    df_clean["Close"] = pd.to_numeric(df_clean["Close"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Close"])
    if df_clean.empty:
        raise ValueError("After normalization, no valid Close data remains.")
    return df_clean


# ==============================
# PAGE SETUP
# ==============================
st.set_page_config(page_title="Market Regime Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; }
.card {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.card-title {
  font-size: 0.9rem;
  letter-spacing: 0.08em;
  opacity: 0.75;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.big { font-size: 1.35rem; font-weight: 700; }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
.hr { height: 1px; background: rgba(255,255,255,0.08); margin: 14px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.markdown("## ⚡ Market Regime Engine")
st.sidebar.caption("Crypto-only MVP • Dark terminal UI")
st.sidebar.caption("DASHBOARD VERSION: 2026-01-09 v5.3 (Normalization fix)")

st.sidebar.markdown("### Market")
_ = st.sidebar.selectbox("Universe", ["Crypto"], index=0, disabled=True)

SUPPORTED_ASSETS = {
    "BTC-USD": "Bitcoin (BTC)",
    "ETH-USD": "Ethereum (ETH)",
    "SOL-USD": "Solana (SOL)",
}

st.sidebar.markdown("### Asset")
ticker = st.sidebar.selectbox(
    "Select asset",
    list(SUPPORTED_ASSETS.keys()),
    format_func=lambda k: SUPPORTED_ASSETS.get(k, k),
    index=0,
)
st.sidebar.caption("Supported assets (MVP): BTC, ETH, SOL")

st.sidebar.markdown("### Strategy")
mode = st.sidebar.selectbox("Risk profile", ["balanced", "conservative", "ma_only"], index=0)
refresh_minutes = st.sidebar.slider("Auto-refresh (minutes)", 1, 30, 5)

st.sidebar.markdown("### Alerts")
enable_alerts = st.sidebar.toggle("Enable Discord alerts", value=False)
cooldown_minutes = st.sidebar.slider("Alert cooldown (minutes)", 1, 120, 15)
cooldown_s = int(cooldown_minutes) * 60

if "discord_test_status" not in st.session_state:
    st.session_state.discord_test_status = None

with st.sidebar.expander("Advanced / Setup"):
    if st.button("Send test Discord alert", use_container_width=True, key="discord_test_btn"):
        st.session_state.discord_test_status = ("pending", "Sending test alert...")
        webhook = st.secrets.get("DISCORD_WEBHOOK_URL", "")
        if not webhook:
            st.session_state.discord_test_status = ("fail", "DISCORD_WEBHOOK_URL secret is missing/empty.")
        else:
            ok, msg = send_discord_alert(
                webhook_url=webhook,
                content="✅ Test: Discord alerts are working! (Market Regime Engine)"
            )
            st.session_state.discord_test_status = ("ok", msg) if ok else ("fail", msg)

    status = st.session_state.discord_test_status
    if status:
        kind, msg = status
        if kind == "pending":
            st.info(msg)
        elif kind == "ok":
            st.success(msg)
        else:
            st.error(msg)

    st.caption("Webhook loaded: " + ("YES ✅" if st.secrets.get("DISCORD_WEBHOOK_URL", "") else "NO ❌"))

st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")


# ==============================
# MODE THRESHOLDS
# ==============================
if mode == "conservative":
    trend_thr = 0.75
    neutral_thr = 0.50
elif mode == "balanced":
    trend_thr = 0.50
    neutral_thr = 0.50
else:
    trend_thr = None
    neutral_thr = None


# ==============================
# MODEL
# ==============================
ensure_model_exists("BTC-USD")
clf, scaler = load_model()


# ==============================
# HISTORY
# ==============================
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []


# ==============================
# LOAD + NORMALIZE DATA (THIS PREVENTS ALL CLOSE ERRORS)
# ==============================
raw = data.get_price_data(ticker)

try:
    df = _normalize_price_df(raw)  # <-- guaranteed Close column
except Exception as e:
    st.error(f"Data normalization failed: {e}")
    st.stop()

# Need enough history for MAs and features
if len(df) < 120:
    st.error("Not enough price history returned to run the engine (need ~120+ points). Try again shortly.")
    st.stop()

# Compute features on a clean base
df = features.compute_features(df)
df = create_labels(df)

# Chart frame
df_price = df.reset_index().rename(columns={"index": "Date"})


# ==============================
# MOVING AVERAGE SIGNAL (GUARDED)
# ==============================
fast = 24
slow = 72

df_ma = df.copy()
df_ma["ma_fast"] = df_ma["Close"].rolling(fast, min_periods=fast).mean()
df_ma["ma_slow"] = df_ma["Close"].rolling(slow, min_periods=slow).mean()
df_ma = df_ma.dropna(subset=["ma_fast", "ma_slow"])

if df_ma.empty:
    st.error(f"Not enough data to compute MAs (need {slow}+ valid points).")
    st.stop()

ma_long = float(df_ma["ma_fast"].iloc[-1] > df_ma["ma_slow"].iloc[-1])


# ==============================
# AI REGIME PROBABILITIES
# ==============================
latest = df.iloc[-1]
X_latest = np.array([float(latest["return"]), float(latest["volatility"]), float(latest["trend"])]).reshape(1, -1)
X_scaled = scaler.transform(X_latest)

probs = clf.predict_proba(X_scaled)[0]
classes = clf.classes_
prob = dict(zip(classes, probs))

p_trend = float(prob.get("Trend", 0.0))
p_neutral = float(prob.get("Unknown", 0.0))  # internal Unknown, user-facing Neutral


# ==============================
# DECISION LOGIC
# ==============================
if mode == "ma_only":
    allow = True
    exposure = ma_long
else:
    allow = (p_trend >= trend_thr) and (p_neutral < neutral_thr)
    exposure = ma_long if allow else 0.0

now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# ==============================
# ALERTS (ENTRY/EXIT ONLY)
# ==============================
alert_state = _load_alert_state()
alert_key_state = f"state::{ticker}::{mode}"
alert_key_sent = f"last_sent::{ticker}::{mode}"

current_state = {"allow": bool(allow), "p_trend": float(p_trend), "p_neutral": float(p_neutral)}
previous_state = alert_state.get(alert_key_state)

if previous_state is None:
    alert_state[alert_key_state] = current_state
    _save_alert_state(alert_state)
else:
    prev_allow = bool(previous_state.get("allow", False))
    entry_event = (prev_allow is False and bool(allow) is True)
    exit_event = (prev_allow is True and bool(allow) is False)

    if enable_alerts and (entry_event or exit_event) and _should_send_with_cooldown(alert_state, alert_key_sent, cooldown_s):
        headline = "🟢 ENTRY" if entry_event else "🔴 EXIT"
        action_line = "Trading conditions have turned favourable." if entry_event else "Trading conditions have deteriorated."

        msg = (
            f"{headline} — {ticker} ({mode})\n"
            f"{action_line}\n\n"
            f"Exposure: {exposure:.2f}\n"
            f"Trend confidence: {p_trend:.2f}\n"
            f"Neutral confidence: {p_neutral:.2f}\n"
            f"Time: {now_utc}"
        )

        webhook = st.secrets.get("DISCORD_WEBHOOK_URL", "")
        ok, info = send_discord_alert(webhook_url=webhook, content=msg)
        if ok:
            _mark_sent(alert_state, alert_key_sent)
        else:
            st.sidebar.error(f"Alert failed: {info}")

    alert_state[alert_key_state] = current_state
    _save_alert_state(alert_state)


# ==============================
# UI (PRO)
# ==============================
asset_name = SUPPORTED_ASSETS.get(ticker, ticker)
tagline = "Market regime filter that helps you avoid trading low-edge conditions."

st.markdown(
    f"""
<div class="card">
  <div class="card-title">MARKET REGIME ENGINE</div>
  <div class="big">{asset_name} <span class="mono" style="opacity:0.7;">({ticker})</span></div>
  <div style="opacity:0.80; margin-top:6px;">{tagline}</div>
  <div class="hr"></div>
  <div style="display:flex; gap:16px; flex-wrap:wrap; opacity:0.85;">
    <div><span class="mono">Mode</span>: <b>{mode}</b></div>
    <div><span class="mono">Last update</span>: <b>{now_utc}</b></div>
    <div><span class="mono">Alerts</span>: <b>{"ON" if enable_alerts else "OFF"}</b></div>
    <div><span class="mono">Cooldown</span>: <b>{cooldown_minutes}m</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

status_text = "ALLOWED ✅" if allow else "BLOCKED ⛔"
if allow and ma_long == 1.0:
    guidance = "Conditions favourable and MA is long — long exposure is permitted."
elif allow and ma_long == 0.0:
    guidance = "Conditions favourable, but MA is not long — no long exposure right now."
else:
    guidance = "Conditions unfavourable — stand aside (no exposure)."

if p_trend >= 0.55:
    context = "Trend conditions detected — directional moves are more likely."
elif p_neutral >= 0.50:
    context = "Neutral / consolidation environment — edge is low (this is normal)."
else:
    context = "Transitional or unstable market — conditions are uncertain."

c1, c2 = st.columns([1.2, 1.0], gap="large")

with c1:
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">Decision</div>
  <div style="display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;">
    <div>
      <div style="opacity:0.75;">Status</div>
      <div class="big">{status_text}</div>
    </div>
    <div>
      <div style="opacity:0.75;">Recommended Exposure</div>
      <div class="big mono">{exposure:.2f}</div>
    </div>
    <div>
      <div style="opacity:0.75;">MA Signal</div>
      <div class="big mono">{ma_long:.0f}</div>
    </div>
  </div>
  <div class="hr"></div>
  <div><b>What should I do?</b> {guidance}</div>
  <div style="margin-top:10px; opacity:0.80;">
    <span class="mono">Trend</span>: <b>{p_trend:.2f}</b> &nbsp;•&nbsp;
    <span class="mono">Neutral</span>: <b>{p_neutral:.2f}</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    gate_line = "MA-only mode (no AI gate)."
    if mode != "ma_only":
        gate_line = f"Gate: Trend ≥ {trend_thr:.2f} AND Neutral < {neutral_thr:.2f}"
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">Market context</div>
  <div style="opacity:0.92; font-size:1.05rem;"><b>{context}</b></div>
  <div class="hr"></div>
  <div style="opacity:0.85;">{gate_line}</div>
  <div style="opacity:0.70; margin-top:10px;">
    This tool is decision support, not financial advice.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")

ch1, ch2 = st.columns([1.35, 1.0], gap="large")

with ch1:
    st.markdown('<div class="card"><div class="card-title">Price (recent)</div>', unsafe_allow_html=True)
    price_tail = df_price.tail(300).copy()

    price_chart = (
        alt.Chart(price_tail)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=""),
            y=alt.Y("Close:Q", title="Price"),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Close:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(price_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ch2:
    st.markdown('<div class="card"><div class="card-title">Regime probabilities (now)</div>', unsafe_allow_html=True)
    probs_df = pd.DataFrame({"Regime": classes, "Probability": probs}).sort_values("Probability", ascending=False)
    probs_df["Regime"] = probs_df["Regime"].replace({"Unknown": "Neutral"})

    bar = (
        alt.Chart(probs_df)
        .mark_bar()
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title=""),
            y=alt.Y("Regime:N", sort="-x", title=""),
            tooltip=[alt.Tooltip("Regime:N"), alt.Tooltip("Probability:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(bar, use_container_width=True)
    st.markdown('<div style="opacity:0.75; margin-top:8px;">Neutral is common — often consolidation / low edge.</div></div>', unsafe_allow_html=True)
