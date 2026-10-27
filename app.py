"""Pre-market command center.  Run:  streamlit run app.py"""
from __future__ import annotations

import datetime as dt
import os
import importlib.util
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import data
import patterns as pt
import session as ses
import uw

ROOT = Path(__file__).parent
UP, DOWN = "#0ca30c", "#d03b3b"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order: blue, orange, aqua
MUTED = "#898781"

COMPANY = "Wiggins Investments LLC"
OPERATOR = "Sa'Rah El"

st.set_page_config(page_title=f"{COMPANY} — Trading Terminal", page_icon="📈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.6rem; }
.hdr { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;
       background:#111827; border:1px solid #1f2937; border-radius:12px; padding:12px 16px; margin-bottom:8px; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { background:#2563eb; border-radius:10px; padding:6px 9px; font-size:20px; box-shadow:0 4px 14px rgba(59,130,246,.3); }
.name { font-weight:700; letter-spacing:.04em; color:#fff; font-size:17px; }
.badge { font: 600 10px ui-monospace, monospace; color:#60a5fa; background:rgba(30,58,138,.6);
         border:1px solid rgba(29,78,216,.5); border-radius:999px; padding:2px 8px; margin-left:6px; vertical-align:middle; }
.sub { color:#94a3b8; font-size:12px; }
.status { display:flex; gap:8px; flex-wrap:wrap; }
.pill { background:#0f172a; border:1px solid #1f2937; border-radius:8px; padding:5px 10px;
        font: 12px ui-monospace, monospace; color:#cbd5e1; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.dot.ok { background:#10b981; animation:pulse 1.6s infinite; } .dot.off { background:#f59e0b; }
b.ok { color:#34d399; } b.off { color:#fbbf24; }
@keyframes pulse { 50% { opacity:.35; } }
[data-testid="stMetric"] { background:#111827; border:1px solid #1f2937; border-radius:12px; padding:10px 14px; }
[data-testid="stDataFrame"] { border:1px solid #1f2937; border-radius:12px; }
@media (max-width: 640px) {
  .block-container { padding-left:.6rem; padding-right:.6rem; }
  .name { font-size:14px; } .badge { display:none; }
  [data-testid="stMetricValue"] { font-size:1.3rem; }
  h2, h3 { font-size:1.1rem !important; }
  .st-key-macro [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:8px; }
  .st-key-macro [data-testid="stColumn"] { min-width:calc(50% - 8px) !important; flex:1 1 calc(50% - 8px) !important; }
  [data-testid="stMetric"] { padding:6px 10px; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
default_wl = (ROOT / "watchlist.txt").read_text().split() if (ROOT / "watchlist.txt").exists() else ["SPY"]
with st.sidebar:
    st.header("Settings")
    wl_text = st.text_area("Watchlist (one per line)", "\n".join(default_wl), height=220)
    tickers = [t.strip().upper() for t in wl_text.replace(",", "\n").split() if t.strip()]
    demo = st.toggle("Demo data (offline)", value=os.environ.get("DASH_DEMO") == "1",
                     help="Synthetic prices – use to test the layout without internet.")
    data.FORCE_DEMO = demo
    st.subheader("Swing parameters")
    stop_atr = st.slider("Stop (× ATR)", 0.5, 3.0, 1.5, 0.25)
    target_atr = st.slider("Target (× ATR)", 1.0, 6.0, 3.0, 0.25)
    max_days = st.slider("Max holding days", 3, 40, 15)
    rate = st.number_input("Risk-free rate", 0.0, 0.10, 0.04, 0.005, format="%.3f")
    st.subheader("Live mode")
    live = st.toggle("Auto-refresh", value=False,
                     help="Re-pulls pre-market quotes and Unusual Whales panels on a timer.")
    refresh_s = st.select_slider("Every", [10, 15, 30, 60, 120], value=30,
                                 format_func=lambda s: f"{s}s", disabled=not live)
    run_every = refresh_s if live else None
    if st.button("🔄 Refresh all data", width="stretch"):
        st.cache_data.clear()
        data.WARNINGS.clear()
    st.caption("Prices: Yahoo Finance (can be delayed). Flow: Unusual Whales (with key). "
               "Not financial advice.")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=10, show_spinner=False)
def load_premarket(tks: tuple, demo: bool):
    return data.premarket_snapshot(list(tks), demo=demo)


@st.cache_data(ttl=30, show_spinner=False)
def load_macro(demo: bool):
    return data.macro_snapshot(demo=demo)


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily(t: str, demo: bool):
    return data.daily_history(t, demo=demo)


@st.cache_data(ttl=10, show_spinner=False)
def load_intraday(t: str, demo: bool, interval="5m", days=5):
    return data.intraday(t, days=days, interval=interval, demo=demo)


@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings(tks: tuple, demo: bool):
    return pd.DataFrame([data.next_earnings(t, demo=demo) for t in tks])


@st.cache_data(ttl=900, show_spinner=False)
def load_chain(t: str, expiry: str, spot: float, demo: bool):
    return data.option_chain(t, expiry, spot, demo=demo)


@st.cache_data(ttl=3600, show_spinner=False)
def load_expirations(t: str, demo: bool):
    return data.expirations(t, demo=demo)


@st.cache_data(ttl=3600, show_spinner=False)
def load_swing(tks: tuple, demo: bool, stop_atr, target_atr, max_days):
    return [an.swing_scan(t, data.daily_history(t, demo=demo), stop_atr, target_atr, max_days)
            for t in tks]


@st.cache_data(ttl=10, show_spinner=False)
def load_today(tks: tuple, demo: bool) -> pd.DataFrame:
    return pd.DataFrame([ses.today(t, data.intraday(t, days=5, interval="5m", demo=demo)) for t in tks])


@st.cache_data(ttl=3600, show_spinner=False)
def load_session_history(t: str, demo: bool) -> pd.DataFrame:
    return ses.history(data.intraday(t, days=60, interval="5m", demo=demo))


@st.cache_data(ttl=3600, show_spinner=False)
def load_patterns(tks: tuple, demo: bool, timeframe: str):
    found, stats = [], {}
    for t in tks:
        if timeframe == "Daily":
            df = data.daily_history(t, demo=demo)
        else:
            df = data.intraday(t, days=60, interval="5m", demo=demo)
            m = df.index.hour * 60 + df.index.minute
            df = df[(m >= 570) & (m < 960)]
        if len(df) < 60:
            continue
        today_found, st_df = pt.scan(t, df)
        found += today_found
        stats[t] = st_df
    return found, stats


@st.cache_data(ttl=8, show_spinner=False)
def load_uw(name: str, params: tuple = (), ticker: str | None = None):
    return uw.fetch(name, dict(params), ticker)


def uw_panel(name: str, params: dict | None = None, ticker: str | None = None) -> pd.DataFrame | None:
    """Fetch a UW endpoint; show a warning instead of crashing on errors."""
    try:
        return load_uw(name, tuple(sorted((params or {}).items())), ticker)
    except uw.UWError as e:
        st.warning(str(e))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Unusual Whales request failed: {e}")
    return None


def first_col(df: pd.DataFrame, *names):
    return next((n for n in names if n in df.columns), None)


def last_price(t: str) -> float:
    i = load_intraday(t, demo, days=2)
    return float(i["Close"].iloc[-1]) if len(i) else float(load_daily(t, demo)["Close"].iloc[-1])


def bias_color(v):
    return f"color: {UP}" if v in ("BULLISH", "Bullish") else f"color: {DOWN}" if v in ("BEARISH", "Bearish") else ""


def signed_color(v):
    if pd.isna(v):
        return ""
    return f"color: {UP}" if v > 0 else f"color: {DOWN}" if v < 0 else ""


def candles(df: pd.DataFrame, name="Price") -> go.Candlestick:
    return go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                          close=df["Close"], name=name, increasing_line_color=UP,
                          decreasing_line_color=DOWN, increasing_line_width=1,
                          decreasing_line_width=1)


def style_fig(fig: go.Figure, height=480) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=0))
    return fig


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
now = data.now_et()
status = data.session_status(now)
open_dt = now.replace(hour=9, minute=30, second=0, microsecond=0)
if status == "Pre-market":
    mins = int((open_dt - now).total_seconds() // 60)
    status_note = f"opens in {mins // 60}h {mins % 60}m"
else:
    status_note = ""
feed_ok = bool(uw.api_key()) and not demo
feed = ('<span class="dot ok"></span>UW FEED: <b class="ok">CONNECTED</b>' if feed_ok else
        '<span class="dot off"></span>UW FEED: <b class="off">DEMO DATA</b>' if demo else
        '<span class="dot off"></span>UW FEED: <b class="off">NO KEY</b>')
st.markdown(f"""
<div class="hdr">
  <div class="brand">
    <div class="logo">📈</div>
    <div>
      <div class="name">{COMPANY.upper()} <span class="badge">LIVE TERMINAL</span></div>
      <div class="sub">Operator: {OPERATOR}</div>
    </div>
  </div>
  <div class="status">
    <div class="pill">{feed}</div>
    <div class="pill">NY {now:%a %H:%M} · {status}{(" · " + status_note) if status_note else ""}</div>
  </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🌅 Pre-Market", "🧭 PM → Close", "📐 Setups & Patterns", "🐋 Options Flow",
                "🎯 Options & LEAPS", "📅 Earnings", "📈 Swing Odds", "⚡ Day Trading",
                "🧩 My Scripts"])

# --------------------------------------------------------------------------- #
# 1. Pre-market
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Futures & macro")
    with st.spinner("Loading futures…"):
        macro = load_macro(demo)
    cols = st.container(key="macro").columns(5)
    for i, r in macro.iterrows():
        fmt = f"{r['Last']:,.2f}" if r["Last"] < 1000 else f"{r['Last']:,.0f}"
        cols[i % 5].metric(r["Name"], fmt, f"{r['Chg %']:+.2f}%" if pd.notna(r["Chg %"]) else None,
                           delta_color="inverse" if r["Symbol"] in ("^VIX", "^TNX") else "normal")

    @st.fragment(run_every=run_every)
    def watchlist_gaps():
        st.subheader("Watchlist gaps")
        with st.spinner("Loading pre-market data…"):
            pm = load_today(tuple(tickers), demo)
        cols_pm = [c for c in ("Ticker", "Prev close", "PM last", "Gap %", "PM high", "PM low",
                               "PM VWAP", "PM bias", "PM volume", "Last PM print", "Behavior")
                   if c in pm.columns]
        pm = pm[cols_pm]
        if "Gap %" in pm:
            pm = pm.reindex(pm["Gap %"].abs().sort_values(ascending=False, na_position="last").index)
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(
                pm.style.format({"Prev close": "{:,.2f}", "PM last": "{:,.2f}", "PM high": "{:,.2f}",
                                 "PM low": "{:,.2f}", "PM VWAP": "{:,.2f}", "PM volume": "{:,.0f}",
                                 "Gap %": "{:+.2f}%"}, na_rep="—")
                .map(signed_color, subset=[c for c in ["Gap %"] if c in pm.columns])
                .map(bias_color, subset=[c for c in ["PM bias"] if c in pm.columns]),
                hide_index=True, width="stretch", height=min(38 * (len(pm) + 1), 620))
            missing = pm[pm["Behavior"] == "no pre-market trades"]["Ticker"].tolist() if "Behavior" in pm else []
            if missing:
                st.info(f"No pre-market trades yet for: {', '.join(missing)}. Thinly traded ETFs "
                        "often don't print until ~8:00 ET – the row will fill once they trade.")
        with right:
            g = (pm.dropna(subset=["Gap %"]).sort_values("Gap %") if "Gap %" in pm
                 else pd.DataFrame(columns=["Ticker", "Gap %"]))
            fig = go.Figure(go.Bar(x=g["Gap %"], y=g["Ticker"], orientation="h",
                                   marker_color=[UP if v > 0 else DOWN for v in g["Gap %"]],
                                   hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
            fig.update_layout(title="Gap % vs prior close", height=max(300, 26 * len(g)),
                              margin=dict(l=10, r=10, t=40, b=10), xaxis_ticksuffix="%")
            st.plotly_chart(fig, width="stretch", theme="streamlit")

        if live:
            st.caption(f"Live · updated {data.now_et():%H:%M:%S} ET")

    watchlist_gaps()

    pick = st.selectbox("Pre-market chart", tickers, key="pm_pick")
    intra = load_intraday(pick, demo, days=2)
    if len(intra):
        today = intra[intra.index.date == intra.index.date[-1]]
        fig = go.Figure(candles(today, pick))
        fig.add_vrect(x0=today.index[0], x1=today.index[0].replace(hour=9, minute=30),
                      fillcolor=MUTED, opacity=0.08, line_width=0,
                      annotation_text="pre-market", annotation_position="top left")
        st.plotly_chart(style_fig(fig, 420), width="stretch", theme="streamlit")

# --------------------------------------------------------------------------- #
# 1a. Pre-market setup vs what happened by the afternoon
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.caption("Your pre-market rules: **PM bias** = last pre-market price vs pre-market VWAP. "
               "**Opening range** = 9:30–10:00. **Behavior** compares the price after 10:00 with "
               "the opening range. **Afternoon** = 1:00–4:00 PM.")

    @st.fragment(run_every=run_every)
    def pm_to_close():
        with st.spinner("Loading today's sessions…"):
            td = load_today(tuple(tickers), demo)
        cols = [c for c in ("Ticker", "PM bias", "Gap %", "PM high", "PM low", "PM VWAP",
                            "ORB high", "ORB low", "Price", "Behavior", "Took PM high", "Took PM low",
                            "Morning %", "Afternoon %", "Afternoon followed PM?", "Close vs PM VWAP")
                if c in td.columns]
        st.subheader("Today")
        st.dataframe(
            td[cols].style.format({c: "{:,.2f}" for c in ("PM high", "PM low", "PM VWAP", "ORB high",
                                                           "ORB low", "Price") if c in cols}
                                  | {c: "{:+.2f}%" for c in ("Gap %", "Morning %", "Afternoon %")
                                     if c in cols}, na_rep="—")
            .map(bias_color, subset=["PM bias"] if "PM bias" in cols else [])
            .map(signed_color, subset=[c for c in ("Gap %", "Morning %", "Afternoon %") if c in cols]),
            hide_index=True, width="stretch", height=min(38 * (len(td) + 1), 700))
        if live:
            st.caption(f"Live · updated {data.now_et():%H:%M:%S} ET")

    pm_to_close()

    st.subheader("History – how each ticker's pre-market setup usually plays out")
    st.caption("Last ~60 sessions of 5-minute data. Continuation = bullish PM → closes above the "
               "opening range (or bearish → below).")
    if st.toggle("Compute history for the whole watchlist", value=False,
                 help="Takes a few seconds per ticker the first time, then it's cached for an hour."):
        rows = []
        prog = st.progress(0.0)
        for i, t in enumerate(tickers):
            h = load_session_history(t, demo)
            for b in ("BULLISH", "BEARISH"):
                o = ses.odds(h, b)
                if o:
                    rows.append({"Ticker": t, "PM bias": b, **o})
            prog.progress((i + 1) / len(tickers))
        prog.empty()
        if rows:
            od = pd.DataFrame(rows)
            st.dataframe(od.style.format({c: "{:.0f}%" for c in od.columns if c.endswith("%")})
                         .map(bias_color, subset=["PM bias"]), hide_index=True, width="stretch")

    ht = st.selectbox("Day-by-day for one ticker", tickers, key="hist_pick")
    h = load_session_history(ht, demo)
    if h.empty:
        st.info("No completed sessions found.")
    else:
        c1, c2 = st.columns(2)
        for col, b in ((c1, "BULLISH"), (c2, "BEARISH")):
            o = ses.odds(h, b)
            with col:
                st.markdown(f"**When {ht}'s pre-market was {b.lower()}** ({o.get('Sessions', 0)} sessions)")
                if o:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Continued", f"{o['Continuation %']:.0f}%")
                    m2.metric("Afternoon followed", f"{o.get('Afternoon follows PM %', float('nan')):.0f}%")
                    m3.metric("Chopped", f"{o['Chop %']:.0f}%")
        counts = (h.groupby(["PM bias", "Behavior"]).size().rename("Days").reset_index())
        fig = go.Figure()
        for b, color in (("BULLISH", UP), ("BEARISH", DOWN)):
            sub_c = counts[counts["PM bias"] == b]
            fig.add_trace(go.Bar(y=sub_c["Behavior"], x=sub_c["Days"], name=f"PM {b.lower()}",
                                 orientation="h", marker_color=color,
                                 hovertemplate="%{y}: %{x} days<extra></extra>"))
        fig.update_layout(barmode="group", title=f"{ht}: what happened after each pre-market bias",
                          height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch", theme="streamlit")
        show = [c for c in ("Date", "PM bias", "Gap %", "Behavior", "Took PM high", "Took PM low",
                            "Morning %", "Afternoon %", "Afternoon followed PM?") if c in h.columns]
        st.dataframe(h[show].iloc[::-1].style.format({"Gap %": "{:+.2f}%", "Morning %": "{:+.2f}%",
                                                      "Afternoon %": "{:+.2f}%"}, na_rep="—")
                     .map(bias_color, subset=["PM bias"]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# 1b. Chart patterns / setups
# --------------------------------------------------------------------------- #
with tabs[2]:
    c1, c2, c3 = st.columns([1, 1, 2])
    tf = c1.radio("Timeframe", ["Daily", "5-minute"], horizontal=True,
                  help="Daily = swing setups. 5-minute = intraday setups (regular hours, last 60 days).")
    bias_f = c2.multiselect("Bias", ["Bullish", "Bearish", "Neutral"],
                            default=["Bullish", "Bearish", "Neutral"])
    with st.spinner("Scanning patterns…"):
        found, pstats = load_patterns(tuple(tickers), demo, tf)
    horizon_col = "Worked % (10d)"
    fd = pd.DataFrame(found)
    if fd.empty:
        st.info("No patterns on the latest bar for your watchlist.")
    else:
        fd = fd[fd["Bias"].isin(bias_f)]
        summary = (pd.DataFrame(found).groupby("Ticker")
                   .apply(lambda g: pt.overall_bias(g.to_dict("records")), include_groups=False)
                   .rename("Overall"))
        fd = fd.join(summary, on="Ticker")
        # Neutral patterns have no "worked" direction: show a dash instead of a blank.
        fd[horizon_col] = pd.to_numeric(fd[horizon_col], errors="coerce").map(
            lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
        fd["Avg move % (10d)"] = pd.to_numeric(fd["Avg move % (10d)"], errors="coerce")
        bars = "bars" if tf != "Daily" else "days"
        st.caption(f"**Worked %** = how often price moved in the pattern's direction over the next 10 "
                   f"{bars}, the times this pattern showed up before on that same ticker. "
                   "Under ~10 past occurrences, treat it as anecdotal.")
        show = ["Ticker", "Overall", "Pattern", "Bias", "Bars in pattern", "Times seen", horizon_col,
                "Avg move % (10d)"]
        st.dataframe(fd[show].sort_values(["Overall", "Ticker"]).style
                     .format({"Avg move % (10d)": "{:+.2f}%"}, na_rep="—")
                     .map(bias_color, subset=["Overall", "Bias"]),
                     hide_index=True, width="stretch", height=min(38 * (len(fd) + 1), 600))

    pk = st.selectbox("Chart", tickers, key="pat_pick")
    if tf == "Daily":
        cd = data.daily_history(pk, demo=demo).tail(150)
    else:
        cd = data.intraday(pk, days=5, interval="5m", demo=demo)
        m = cd.index.hour * 60 + cd.index.minute
        cd = cd[(m >= 570) & (m < 960)]
    mine = [f for f in found if f["Ticker"] == pk]
    fig = go.Figure([candles(cd, pk)])
    for f in mine:
        for x0, y0, x1, y1, lbl in pt.pattern_lines(cd, f["Pattern"]):
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", name=f"{f['Pattern']} ({lbl})",
                                     line=dict(color=SERIES[0] if lbl == "upper" else SERIES[1],
                                               width=2, dash="dash")))
    if tf != "Daily":
        fig.update_xaxes(rangebreaks=[dict(bounds=[16, 9.5], pattern="hour"), dict(bounds=["sat", "mon"])])
    else:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(style_fig(fig, 480), width="stretch", theme="streamlit")
    if mine:
        st.write("Present now: " + ", ".join(f"**{f['Pattern']}** ({f['Bias'].lower()})" for f in mine))
    if pk in pstats:
        with st.expander(f"All patterns – {pk} history"):
            ps = pstats[pk].copy()
            ps[horizon_col] = ps[horizon_col].map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
            st.dataframe(ps.style.format({"Avg move % (10d)": "{:+.2f}%",
                                                  "Avg abs move %": "{:.2f}%"}, na_rep="—"),
                         hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# 1c. Unusual Whales (real-time flow, dark pool, GEX, market tide)
# --------------------------------------------------------------------------- #
with tabs[3]:
    if not uw.api_key():
        st.info("**Connect Unusual Whales:** on the hosted site, open the app's **Settings → Secrets** "
                "and add `UW_API_KEY = \"your_key\"`. Running on your Mac, it reads the key from "
                "`~/TradingJournal/uw_config.json` automatically.")
    else:
        sub = st.tabs(["Flow alerts", "Ticker options", "Market tide", "Dark pool", "Gamma exposure",
                       "Earnings today", "Any endpoint"])

        with sub[1]:
            ot = st.selectbox("Ticker", tickers, key="uw_ticker")

            @st.fragment(run_every=run_every)
            def ticker_options():
                c1, c2, c3 = st.columns(3)
                iv = uw_panel("iv_rank", ticker=ot)
                if iv is not None and len(iv):
                    col = first_col(iv, "iv_rank_1y", "iv_rank", "rank")
                    if col:
                        c1.metric("IV rank (1y)", f"{float(iv[col].iloc[-1]):.0f}")
                mp = uw_panel("max_pain", ticker=ot)
                if mp is not None and len(mp):
                    col = first_col(mp, "max_pain")
                    if col:
                        c2.metric("Max pain (nearest expiry)", f"{float(mp[col].iloc[0]):,.2f}")
                c3.metric("Price", f"{last_price(ot):,.2f}")
                st.markdown("**Today's options volume & premium**")
                ov = uw_panel("options_volume", ticker=ot)
                if ov is not None and len(ov):
                    st.dataframe(ov.head(5), hide_index=True, width="stretch")
                st.markdown(f"**Flow alerts – {ot}**")
                fa = uw_panel("flow_alerts", {"ticker_symbol": ot, "limit": 100})
                if fa is not None and len(fa):
                    st.dataframe(fa, hide_index=True, width="stretch", height=380)
                else:
                    st.write("No flow alerts for this ticker today.")
                st.markdown(f"**Recent trades – {ot}**")
                fr = uw_panel("flow_ticker", ticker=ot)
                if fr is not None and len(fr):
                    st.dataframe(fr.head(200), hide_index=True, width="stretch", height=380)

            ticker_options()

        with sub[0]:
            c1, c2, c3, c4 = st.columns(4)
            scope = c1.radio("Tickers", ["Watchlist", "All", "One"], horizontal=True)
            one = c2.text_input("Ticker", "SPY", disabled=scope != "One").upper()
            min_prem = c3.number_input("Min premium $", 0, 10_000_000, 100_000, 50_000)
            limit = c4.slider("Rows", 50, 500, 200, 50)

            @st.fragment(run_every=run_every)
            def flow_alerts():
                params = {"min_premium": min_prem, "limit": limit}
                if scope == "One":
                    params["ticker_symbol"] = one
                df = uw_panel("flow_alerts", params)
                if df is None or df.empty:
                    st.write("No flow alerts for these filters.")
                    return
                tcol = first_col(df, "ticker", "ticker_symbol", "underlying_symbol")
                if scope == "Watchlist" and tcol:
                    df = df[df[tcol].isin(tickers)]
                pcol = first_col(df, "total_premium", "premium")
                kcol = first_col(df, "type", "option_type")
                if tcol and pcol and kcol and len(df):
                    agg = (df.assign(side=df[kcol].astype(str).str.lower())
                           .pivot_table(index=tcol, columns="side", values=pcol, aggfunc="sum",
                                        fill_value=0))
                    agg["net"] = agg.get("call", 0) - agg.get("put", 0)
                    agg = agg.sort_values("net")
                    fig = go.Figure(go.Bar(
                        x=agg["net"], y=agg.index, orientation="h",
                        marker_color=[UP if v > 0 else DOWN for v in agg["net"]],
                        hovertemplate="%{y}: net call−put premium $%{x:,.0f}<extra></extra>"))
                    fig.update_layout(title="Net call − put premium in these alerts",
                                      height=max(260, 24 * len(agg)),
                                      margin=dict(l=10, r=10, t=40, b=10), xaxis_tickprefix="$")
                    st.plotly_chart(fig, width="stretch", theme="streamlit")
                lead = [c for c in ("created_at", tcol, kcol, "strike", "expiry", pcol,
                                    "total_size", "volume", "open_interest", "underlying_price",
                                    "alert_rule", "has_sweep") if c and c in df.columns]
                st.dataframe(df[lead + [c for c in df.columns if c not in lead]],
                             hide_index=True, width="stretch", height=480)
                st.caption(f"{len(df)} alerts · updated {data.now_et():%H:%M:%S} ET")

            flow_alerts()

        with sub[2]:
            @st.fragment(run_every=run_every)
            def market_tide():
                df = uw_panel("market_tide")
                if df is None or df.empty:
                    return
                xcol = first_col(df, "timestamp", "date", "time")
                ys = [c for c in ("net_call_premium", "net_put_premium") if c in df.columns]
                if xcol and ys:
                    fig = go.Figure()
                    for col, color in zip(ys, SERIES):
                        fig.add_trace(go.Scatter(x=pd.to_datetime(df[xcol]), y=df[col],
                                                 name=col.replace("_", " "),
                                                 line=dict(color=color, width=2)))
                    fig.update_layout(title="Market tide – net premium flowing into calls vs puts",
                                      yaxis_tickprefix="$")
                    st.plotly_chart(style_fig(fig, 420), width="stretch", theme="streamlit")
                st.dataframe(df.tail(50), hide_index=True, width="stretch")

            market_tide()

        with sub[3]:
            c1, c2 = st.columns([1, 3])
            dp_t = c1.text_input("Ticker (blank = all recent prints)", "").upper()

            @st.fragment(run_every=run_every)
            def dark_pool():
                df = (uw_panel("darkpool_ticker", ticker=dp_t) if dp_t
                      else uw_panel("darkpool_recent", {"limit": 200}))
                if df is None or df.empty:
                    return
                pcol = first_col(df, "premium")
                if pcol:
                    df = df.sort_values(pcol, ascending=False)
                st.dataframe(df, hide_index=True, width="stretch", height=520)

            dark_pool()

        with sub[4]:
            gx = st.selectbox("Ticker", tickers, key="gex_pick")
            df = uw_panel("gex_by_strike", ticker=gx)
            if df is not None and not df.empty:
                scol = first_col(df, "strike", "price")
                cg, pg = first_col(df, "call_gamma", "call_gex"), first_col(df, "put_gamma", "put_gex")
                if scol and cg and pg:
                    df = df.assign(net_gamma=df[cg] + df[pg]).sort_values(scol)
                    spot = last_price(gx)
                    df = df[df[scol].between(spot * 0.85, spot * 1.15)]
                    fig = go.Figure(go.Bar(
                        x=df[scol], y=df["net_gamma"],
                        marker_color=[SERIES[0] if v > 0 else SERIES[1] for v in df["net_gamma"]],
                        hovertemplate="Strike %{x}: %{y:,.0f}<extra></extra>"))
                    fig.add_vline(x=spot, line_dash="dot", line_color=MUTED,
                                  annotation_text=f"spot {spot:,.2f}")
                    fig.update_layout(title=f"{gx} net gamma by strike (blue = positive, orange = negative)",
                                      height=420, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, width="stretch", theme="streamlit")
                st.dataframe(df, hide_index=True, width="stretch")

        with sub[5]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Before the open**")
                df = uw_panel("earnings_premarket")
                if df is not None:
                    st.dataframe(df, hide_index=True, width="stretch")
            with c2:
                st.markdown("**After the close**")
                df = uw_panel("earnings_afterhours")
                if df is not None:
                    st.dataframe(df, hide_index=True, width="stretch")

        with sub[6]:
            st.caption("Call any Unusual Whales endpoint from their docs, e.g. "
                       "`/api/stock/NVDA/flow-recent` or `/api/congress/recent-trades`.")
            path = st.text_input("Path", "/api/stock/SPY/options-volume")
            qs = st.text_input("Query params (key=value&key2=value2)", "")
            if st.button("Fetch"):
                params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
                df = uw_panel(path, params)
                if df is not None:
                    st.dataframe(df, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# 2. Earnings
# --------------------------------------------------------------------------- #
with tabs[5]:
    with st.spinner("Loading earnings calendar…"):
        er = load_earnings(tuple(tickers), demo)
    horizon = st.slider("Show reports in the next N days", 1, 90, 30)
    today_d = data.now_et().date()
    upcoming = er.dropna(subset=["Date"]).copy()
    upcoming["Days away"] = [(d - today_d).days for d in upcoming["Date"]]
    upcoming = upcoming[(upcoming["Days away"] >= 0) & (upcoming["Days away"] <= horizon)]
    upcoming = upcoming.sort_values("Date")
    if upcoming.empty:
        st.info("No watchlist names report in this window.")
    else:
        st.dataframe(upcoming.style.format({"EPS est": "{:.2f}", "Revenue est": "{:,.0f}"},
                                           na_rep="—"), hide_index=True, width="stretch")

        et = st.selectbox("Implied vs historical move", upcoming["Ticker"].tolist())
        edate = upcoming.set_index("Ticker").loc[et, "Date"]
        spot = last_price(et)
        exps = [e for e in load_expirations(et, demo) if pd.Timestamp(e).date() >= edate]
        daily = load_daily(et, demo)
        hist = an.earnings_reactions(daily, data.past_earnings_dates(et, demo=demo))
        c1, c2, c3 = st.columns(3)
        imp = np.nan
        if exps:
            imp = an.implied_move(load_chain(et, exps[0], spot, demo), spot)
            c1.metric(f"Implied move (exp {exps[0]})", f"±{imp:.1f}%" if pd.notna(imp) else "—")
        if len(hist):
            avg = hist["Day move %"].abs().mean()
            c2.metric("Avg historical move", f"±{avg:.1f}%")
            if pd.notna(imp):
                beat = (hist["Day move %"].abs() > imp).mean() * 100
                c3.metric("Past moves bigger than implied", f"{beat:.0f}%",
                          help="Above ~50% → options have tended to underprice this name.")
            fig = go.Figure(go.Bar(x=hist["Report"], y=hist["Day move %"],
                                   marker_color=[UP if v > 0 else DOWN for v in hist["Day move %"]],
                                   hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
            if pd.notna(imp):
                for y in (imp, -imp):
                    fig.add_hline(y=y, line_dash="dash", line_color=MUTED,
                                  annotation_text=f"implied {y:+.1f}%")
            fig.update_layout(title=f"{et}: reaction-day moves", yaxis_ticksuffix="%",
                              height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch", theme="streamlit")

# --------------------------------------------------------------------------- #
# 3. Swing setups
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.caption(f"Stop = {stop_atr}×ATR, target = {target_atr}×ATR, max {max_days} days. "
               "**P(target before stop)** = random-walk math; **P(touch target)** = lognormal "
               "with 20-day realized vol; **Hist win rate** = how often this exact setup hit "
               "target before stop over the last ~5 years on this ticker.")
    with st.spinner("Scanning setups (backtesting each ticker)…"):
        scans = load_swing(tuple(tickers), demo, stop_atr, target_atr, max_days)
    sdf = pd.DataFrame([{k: v for k, v in s.items() if k != "_stats"} for s in scans])
    only_active = st.checkbox("Only tickers with a setup today", value=False)
    view = sdf[sdf["Setups today"] != "—"] if only_active else sdf
    view = view.sort_values(["Hist win rate", "P(touch target)"], ascending=False, na_position="last")
    st.dataframe(
        view.style.format({"Price": "{:,.2f}", "RSI": "{:.0f}", "ATR %": "{:.1f}%",
                           "Entry": "{:,.2f}", "Stop": "{:,.2f}", "Target": "{:,.2f}",
                           "P(target before stop)": "{:.0%}", "P(touch target)": "{:.0%}",
                           "Hist win rate": "{:.0%}"}, na_rep="—"),
        hide_index=True, width="stretch")

    sp = st.selectbox("Chart & setup history", tickers, key="swing_pick")
    sc = next(s for s in scans if s["Ticker"] == sp)
    st.dataframe(pd.DataFrame(sc["_stats"]).T.rename(columns={"n": "Signals", "win_rate": "Win rate",
                                                              "avg_ret": "Avg return"})
                 .style.format({"Win rate": "{:.0%}", "Avg return": "{:+.2%}", "Signals": "{:.0f}"},
                               na_rep="—"), width="stretch")
    d = an.add_indicators(load_daily(sp, demo)).tail(180)
    fig = go.Figure([candles(d, sp)])
    for col, color in zip(("EMA20", "SMA50", "SMA200"), SERIES):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(color=color, width=2)))
    for lbl, y in (("Stop", sc["Stop"]), ("Target", sc["Target"])):
        fig.add_hline(y=y, line_dash="dot", line_color=MUTED, annotation_text=f"{lbl} {y:,.2f}")
    st.plotly_chart(style_fig(fig), width="stretch", theme="streamlit")

# --------------------------------------------------------------------------- #
# 4. LEAPS
# --------------------------------------------------------------------------- #
with tabs[4]:
    c1, c2, c3, c4 = st.columns(4)
    lt = c1.selectbox("Ticker", tickers, key="leaps_pick")
    kind = c2.radio("Type", ["call", "put"], horizontal=True)
    min_dte = c3.number_input("Min days to expiry", 180, 1000, 365, 30)
    vol_src = c4.radio("Probability vol", ["Implied", "Historical (1y)"], horizontal=True,
                       help="Implied = market's view. Historical = what the stock has actually done.")
    spot = last_price(lt)
    daily = load_daily(lt, demo)
    hv = an.realized_vol(daily["Close"], 252) if vol_src.startswith("Hist") else None
    exps = [e for e in load_expirations(lt, demo)
            if (pd.Timestamp(e).date() - data.now_et().date()).days >= min_dte]
    if not exps:
        st.warning(f"No expirations ≥ {min_dte} days for {lt}.")
    else:
        chain = pd.concat([load_chain(lt, e, spot, demo) for e in exps], ignore_index=True)
        tbl = an.leaps_table(chain, spot, rate, kind, hv)
        dmin, dmax = st.slider("Delta range (abs)", 0.05, 0.99, (0.55, 0.95))
        tbl = tbl[tbl["Delta"].abs().between(dmin, dmax)]
        st.metric(f"{lt} spot", f"{spot:,.2f}")
        st.dataframe(
            tbl.style.format({"Strike": "{:,.2f}", "Mid": "{:,.2f}", "IV %": "{:.0f}",
                              "Delta": "{:.2f}", "Theta/day": "{:.3f}", "Breakeven": "{:,.2f}",
                              "BE move %": "{:+.1f}%", "Extrinsic %": "{:.0f}%",
                              "P(ITM)": "{:.0%}", "P(profit @ exp)": "{:.0%}",
                              "Leverage": "{:.1f}x", "Spread %": "{:.1f}%", "Fit score": "{:.0f}"},
                             na_rep="—"),
            hide_index=True, width="stretch", height=480)
        st.caption("**Fit score** favours delta ≈ 0.70–0.85, low extrinsic value (less time decay), "
                   "tight spreads and open interest – the classic stock-replacement LEAPS profile.")

# --------------------------------------------------------------------------- #
# 5. Day trading
# --------------------------------------------------------------------------- #
with tabs[7]:
    c1, c2 = st.columns([1, 3])
    dtk = c1.selectbox("Ticker", tickers, key="dt_pick")
    interval = c1.radio("Bars", ["1m", "5m"], index=1, horizontal=True)
    intra = load_intraday(dtk, demo, interval=interval, days=5 if interval == "5m" else 3)
    info = an.day_levels(intra, load_daily(dtk, demo))
    lv, stx = info["levels"], info["stats"]
    with c1:
        st.metric("ATR (14d)", f"{stx['ATR']:.2f}")
        if "Gap %" in stx:
            st.metric("Gap", f"{stx['Gap %']:+.2f}%")
        if "RVOL" in stx and pd.notna(stx["RVOL"]):
            st.metric("Relative volume", f"{stx['RVOL']:.2f}x")
        if "ATR used %" in stx:
            st.metric("ATR used today", f"{stx['ATR used %']:.0f}%")
        st.dataframe(pd.Series(lv, name="Level").sort_values(ascending=False).map("{:,.2f}".format),
                     width="stretch")
    with c2:
        t = info["today"]
        fig = go.Figure([candles(t, dtk)])
        if len(info["rth"]):
            fig.add_trace(go.Scatter(x=info["rth"].index, y=an.vwap(info["rth"]), name="VWAP",
                                     line=dict(color=SERIES[0], width=2)))
        groups = {"PMH": SERIES[1], "PML": SERIES[1], "ORH": SERIES[2], "ORL": SERIES[2]}
        for k, y in lv.items():
            if k in ("R2", "S2"):
                continue
            fig.add_hline(y=y, line_width=1, line_dash="dot" if k not in groups else "dash",
                          line_color=groups.get(k, MUTED), annotation_text=f"{k} {y:,.2f}",
                          annotation_position="top left", annotation_font_size=11)
        # Frame today's bars plus any key level within half an ATR of them.
        lo, hi = t["Low"].min(), t["High"].max()
        near = [y for y in lv.values() if lo - stx["ATR"] / 2 <= y <= hi + stx["ATR"] / 2]
        lo, hi = min([lo] + near), max([hi] + near)
        pad = (hi - lo) * 0.05
        fig.update_yaxes(range=[lo - pad, hi + pad])
        st.plotly_chart(style_fig(fig, 600), width="stretch", theme="streamlit")
    st.caption("PDH/PDL/PDC = prior day high/low/close · PMH/PML = pre-market high/low · "
               "ORH/ORL = 15-min opening range · R1/S1 = floor pivots. RVOL compares today's "
               "volume so far with the same time of day on prior sessions.")

# --------------------------------------------------------------------------- #
# 6. Plugins: your own scripts
# --------------------------------------------------------------------------- #
with tabs[8]:
    st.caption("Drop a `.py` file into the `plugins/` folder with `TITLE = '…'` and "
               "`def run(tickers): …` – it appears here. See the examples in that folder.")
    plugin_files = sorted(p for p in (ROOT / "plugins").glob("*.py") if not p.name.startswith("_"))
    for p in plugin_files:
        try:
            spec = importlib.util.spec_from_file_location(f"plugin_{p.stem}", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:  # noqa: BLE001
            st.error(f"{p.name} failed to load")
            st.code(traceback.format_exc())
            continue
        with st.expander(getattr(mod, "TITLE", p.stem), expanded=True):
            if st.button("Run", key=f"run_{p.stem}"):
                with st.spinner(f"Running {p.name}…"):
                    try:
                        st.session_state[f"out_{p.stem}"] = mod.run(tickers)
                    except Exception:  # noqa: BLE001
                        st.session_state[f"out_{p.stem}"] = "ERROR\n" + traceback.format_exc()
            out = st.session_state.get(f"out_{p.stem}")
            for item in out if isinstance(out, list) else [out]:
                if item is None:
                    continue
                if isinstance(item, (pd.DataFrame, pd.Series)):
                    st.dataframe(item, width="stretch")
                elif isinstance(item, go.Figure):
                    st.plotly_chart(item, width="stretch", theme="streamlit")
                elif isinstance(item, dict):
                    st.json(item)
                else:
                    st.code(str(item))

if data.WARNINGS:
    with st.sidebar.expander(f"⚠️ {len(data.WARNINGS)} data notices"):
        for w in data.WARNINGS:
            st.write("•", w)
