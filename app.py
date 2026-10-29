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
import levels as lvl
import patterns as pt
import scanner as sc
import sectors as secs
import session as ses
import tvchart
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
_master = (ROOT / "watchlist.txt").read_text() if (ROOT / "watchlist.txt").exists() else "SPY"
default_wl, crypto_list = data.read_watchlist(_master)
with st.sidebar:
    st.header("Settings")
    wl_text = st.text_area("Watchlist (one per line)", "\n".join(default_wl), height=220,
                           help="Every page uses these tickers, except Sectors (S&P 500 list).")
    st.caption(
        "**How to use:** type one ticker per line (e.g. `NVDA`), or separate with commas. "
        "Click outside the box to update every page.  \n"
        "⚠️ Changes here are **temporary** – they reset when the page reloads. "
        "To change the list **permanently**, edit `watchlist.txt` on GitHub "
        "(open the file → ✏️ pencil → edit → **Commit changes**).")
    tickers = data.read_watchlist(wl_text)[0]
    demo = st.toggle("Demo data (offline)", value=os.environ.get("DASH_DEMO") == "1",
                     help="Synthetic prices – use to test the layout without internet.")
    data.FORCE_DEMO = demo
    st.subheader("Swing parameters", help="These settings drive the **📈 Swing Odds** page: where the stop "
                 "and target sit for each setup, and how long a trade may run. Changing them re-tests every "
                 "setup over ~5 years of history, so the win probabilities update.")
    stop_atr = st.slider(
        "Stop (× ATR)", 0.5, 3.0, 1.5, 0.25,
        help="**How far below entry the stop-loss sits**, measured in ATR (the stock's average daily range).\n\n"
             "Example: ATR = \\$2 and Stop = 1.5 → stop is \\$3 below entry.\n\n"
             "**Wider stop** (move right) = fewer stop-outs, so win rates go **up**, but each loss is bigger. "
             "**Tighter stop** (move left) = smaller losses, but more trades get stopped out.")
    target_atr = st.slider(
        "Target (× ATR)", 1.0, 6.0, 3.0, 0.25,
        help="**How far above entry the profit target sits**, in ATR.\n\n"
             "Example: ATR = \\$2 and Target = 3 → target is \\$6 above entry.\n\n"
             "**Closer target** (move left) = reached more often, so win rates go **up**, but each win is "
             "smaller. Target ÷ Stop = your reward-to-risk (3 ÷ 1.5 = 2 : 1).")
    max_days = st.slider(
        "Max holding days", 3, 40, 15,
        help="**How many trading days a swing trade may stay open.** If neither the stop nor the target is "
             "hit in that time, the trade 'times out' and counts as **not a win** in the win probability.\n\n"
             "Longer = more time for the target to be reached (win rates usually go up). "
             "Shorter = faster trades.")
    rate = st.number_input(
        "Risk-free rate", 0.0, 0.10, 0.04, 0.005, format="%.3f",
        help="**Used only on the 🎯 Options & LEAPS page** to price options (the interest you could earn "
             "risk-free, e.g. on Treasury bills). 0.040 = 4%. Small changes barely affect results – "
             "leave it near the current T-bill rate.")
    st.subheader("Live mode")
    live = st.toggle("Auto-refresh", value=False,
                     help="**On** = the live tables and charts (pre-market, key levels, PM → Close, charts, "
                          "options flow) reload by themselves on a timer. **Off** = they update only when you "
                          "reload the page or press 🔄 Refresh all data.")
    refresh_s = st.select_slider("Every", [10, 15, 30, 60, 120], value=30,
                                 format_func=lambda s: f"{s}s", disabled=not live,
                                 help="How often Auto-refresh reloads. 30s is a good balance; very short "
                                      "intervals use more of your Unusual Whales request allowance.")
    run_every = refresh_s if live else None
    st.subheader("Charts")
    candle_style = st.radio("Candle style", ["Filled", "Hollow", "Heikin Ashi"], horizontal=True,
                            key="candle_style",
                            help="Filled = solid candles. Hollow = up candles hollow, down candles filled. "
                                 "Heikin Ashi = smoothed candles that make trends easier to see "
                                 "(their open/close are averages, not actual prices).")
    if st.button("🔄 Refresh all data", width="stretch",
                 help="Throw away everything saved in memory and pull fresh data for every page now."):
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


@st.cache_data(ttl=60, show_spinner=False)
def load_crypto(symbols: tuple, demo: bool) -> pd.DataFrame:
    return data.crypto_snapshot(list(symbols), demo=demo)


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
    out = []
    for t in tks:
        d = data.daily_history(t, demo=demo)
        if len(d) >= 60:   # skip tickers with no (or too little) real data
            out.append(an.swing_scan(t, d, stop_atr, target_atr, max_days))
    return out


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


@st.cache_data(ttl=30, show_spinner=False)
def load_bars(t: str, tf: str, demo: bool, custom: tuple | None = None) -> pd.DataFrame:
    """Candles for a timeframe button, or for a custom (start, end, candle size) range."""
    if tf == "Custom" and custom:
        start, end, candle = custom
        src, rule, _ = lvl.CUSTOM_CANDLES[candle]
        span = None
    else:
        src, rule, span = lvl.TIMEFRAMES[tf]
    if src == "daily":
        df = data.daily_history(t, demo=demo)
    elif src == "dailymax":
        df = data.daily_history(t, period="max", demo=demo)
    else:
        interval, days = {"1m": ("1m", 2), "1m7": ("1m", 7), "5m": ("5m", 5), "5m60": ("5m", 60)}[src]
        df = data.intraday(t, days=days, interval=interval, demo=demo)
    if rule:
        df = lvl.resample(df, rule)
    if len(df):
        df = tvchart.add_indicators(df)   # EMA 9 / EMA 200 / MACD on full history, then trim
    if span is None:
        d = df.index.date
        return df[(d >= start) & (d <= end)]
    return lvl.window(df, span)


def tf_picker(key: str, default: str = "5m") -> tuple[str, tuple | None]:
    """Timeframe buttons shown above every price chart; 'Custom' opens a date range."""
    opts = list(lvl.TIMEFRAMES) + ["Custom"]
    tf = st.radio("Timeframe", opts, index=opts.index(default), horizontal=True, key=key)
    if tf != "Custom":
        return tf, None
    today = data.now_et().date()
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("From", today - dt.timedelta(days=30), key=f"{key}_from", format="DD/MM/YYYY")
    end = c2.date_input("To", today, key=f"{key}_to", format="DD/MM/YYYY")
    candle = c3.selectbox("Candle size", list(lvl.CUSTOM_CANDLES), index=list(lvl.CUSTOM_CANDLES).index("Daily"),
                          key=f"{key}_candle")
    reach = lvl.CUSTOM_CANDLES[candle][2]
    if reach and (today - start).days > reach:
        st.caption(f"ℹ️ {candle} candles only go back about {reach} days – earlier dates will be empty. "
                   "Use 1h, Daily or Weekly for longer ranges.")
    if start > end:
        st.warning("'From' is after 'To' – swap the dates.")
    return tf, (start, end, candle)


def is_intraday(tf: str, custom: tuple | None) -> bool:
    return tf in lvl.INTRADAY or (tf == "Custom" and bool(custom) and custom[2] in lvl.INTRADAY)


@st.cache_data(ttl=30, show_spinner=False)
def load_levels(t: str, demo: bool) -> dict | None:
    """Key levels for a ticker, or None when it has no data."""
    try:
        return lvl.ticker_levels(data.intraday(t, days=5, interval="5m", demo=demo),
                                 data.daily_history(t, demo=demo))
    except ValueError:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_long_sr(t: str, demo: bool, weekly: bool) -> list:
    return lvl.long_term_sr(data.daily_history(t, demo=demo), weekly)


def pretty_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Show date/time columns as 28/Oct/2026 (and 28/Oct/2026 09:45 when there's a time)."""
    df = df.copy()
    for c in df.columns:
        name = str(c).lower()
        if not any(k in name for k in ("date", "time", "_at", "expir", "day_of")) or name in ("time (et)",):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        ts = pd.to_datetime(df[c], errors="coerce", utc=False)
        if ts.notna().sum() == 0 or ts.notna().sum() < df[c].notna().sum():
            continue
        has_time = (ts.dropna().dt.hour.fillna(0) + ts.dropna().dt.minute.fillna(0)).any()
        if has_time and getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("America/New_York")
        df[c] = ts.dt.strftime(lvl.DATETIME_FMT if has_time else lvl.DATE_FMT).fillna("—")
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500() -> pd.DataFrame:
    return secs.sp500()


@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings_week(tks: tuple, start: dt.date, end: dt.date, demo: bool) -> pd.DataFrame:
    return data.earnings_between(list(tks), start, end, demo=demo)


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


@st.cache_data(ttl=30, show_spinner=False)
def load_flow_summary(tks: tuple, min_prem: int) -> pd.DataFrame:
    """Unusual-options-flow totals per ticker (latest session)."""
    frames = []
    try:
        for i in range(0, len(tks), 50):
            frames.append(uw.fetch("flow_alerts", {"ticker_symbol": ",".join(tks[i:i + 50]),
                                                   "min_premium": min_prem, "limit": 200}))
    except uw.UWError:
        # ticker list not accepted: take the most recent alerts market-wide instead
        frames = [uw.fetch("flow_alerts", {"min_premium": min_prem, "limit": 200})]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = sc.flow_summary(df)
    return out[out.index.isin(tks)]


@st.cache_data(ttl=120, show_spinner=False)
def load_gamma(t: str) -> dict:
    return sc.gamma_levels(uw.fetch("spot_gex_strike", ticker=t))


def first_col(df: pd.DataFrame, *names):
    return next((n for n in names if n in df.columns), None)


def last_price(t: str) -> float:
    """Latest price, or NaN when the ticker has no data."""
    i = load_intraday(t, demo, days=2)
    if len(i):
        return float(i["Close"].iloc[-1])
    d = load_daily(t, demo)
    return float(d["Close"].iloc[-1]) if len(d) else float("nan")


def no_data(t: str) -> None:
    st.warning(f"**{t}**: no data – check the symbol (it may be misspelled or not covered by the data feed).")


def usd_signed(v) -> str:
    """+$1.25 / -$0.40 / — for missing."""
    return "—" if pd.isna(v) else f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def bias_color(v):
    return f"color: {UP}" if v in ("BULLISH", "Bullish") else f"color: {DOWN}" if v in ("BEARISH", "Bearish") else ""


def signed_color(v):
    if pd.isna(v):
        return ""
    return f"color: {UP}" if v > 0 else f"color: {DOWN}" if v < 0 else ""


def candles(df: pd.DataFrame, name="Price") -> go.Candlestick:
    """Candles in the style picked in the sidebar (Filled / Hollow / Heikin Ashi)."""
    if candle_style == "Heikin Ashi":
        df = an.heikin_ashi(df)
        name = f"{name} (Heikin Ashi)"
    up_fill = "rgba(0,0,0,0)" if candle_style == "Hollow" else UP
    return go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                          close=df["Close"], name=name, increasing_line_color=UP,
                          decreasing_line_color=DOWN, increasing_fillcolor=up_fill,
                          decreasing_fillcolor=DOWN, increasing_line_width=1,
                          decreasing_line_width=1)


def style_fig(fig: go.Figure, height=480, intraday: bool | None = None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=0))
    if intraday is True:
        fig.update_xaxes(tickformat="%H:%M<br>%d/%b/%Y", hoverformat=lvl.DATETIME_FMT)
    elif intraday is False:
        fig.update_xaxes(tickformat=lvl.DATE_FMT, hoverformat=lvl.DATE_FMT)
    return fig


HIGH_PROB = 0.65      # Swing Odds: "high probability" cut-off
MIN_SIGNALS = 10      # ...and how many past occurrences that needs to count

LEVEL_STYLE = {
    "PMH": ("Pre-market high", SERIES[1], "dash"), "PML": ("Pre-market low", SERIES[1], "dash"),
    "ORH": ("Opening range high", SERIES[2], "dash"), "ORL": ("Opening range low", SERIES[2], "dash"),
    "PDH": ("Prior day high", MUTED, "dot"), "PDL": ("Prior day low", MUTED, "dot"),
    "PDC": ("Prior day close", MUTED, "dot"), "R1": ("Resistance 1 (pivot)", MUTED, "dashdot"),
    "S1": ("Support 1 (pivot)", MUTED, "dashdot"),
}
# Legend for the Day Trading levels table: key -> (full name, what it means for a trade)
LEVEL_INFO = {
    "PMH": ("Pre-market high", "Highest price 4:00–9:29 AM. Often resistance after the open; a break above it is bullish."),
    "PML": ("Pre-market low", "Lowest price 4:00–9:29 AM. Often support; a break below it is bearish."),
    "ORH": ("Opening range high", "Highest price 9:30–10:00. A move above it after 10:00 = breakout (bullish)."),
    "ORL": ("Opening range low", "Lowest price 9:30–10:00. A move below it after 10:00 = breakdown (bearish)."),
    "PDH": ("Prior day high", "Yesterday's high. Common resistance; clearing it shows strength."),
    "PDL": ("Prior day low", "Yesterday's low. Common support; losing it shows weakness."),
    "PDC": ("Prior day close", "Yesterday's close. Today's gap is measured from here; gaps often drift back toward it."),
    "Pivot": ("Pivot point", "(Yesterday's high + low + close) ÷ 3. Trading above it leans bullish, below leans bearish."),
    "R1": ("Resistance 1 (pivot)", "First upside level from pivot math – a common first profit target for longs."),
    "R2": ("Resistance 2 (pivot)", "Second, stronger upside level – reached on strong trend days."),
    "S1": ("Support 1 (pivot)", "First downside level from pivot math – a common first target for shorts."),
    "S2": ("Support 2 (pivot)", "Second, stronger downside level – reached on strong selling days."),
}
LEVEL_GROUPS = {
    "Pre-market high/low": ["PMH", "PML"],
    "Opening range 9:30–10:00": ["ORH", "ORL"],
    "Prior day high/low/close": ["PDH", "PDL", "PDC"],
    "Pivots": ["R1", "S1"],
    "Support / resistance": [],
}


def price_chart(t: str, tf: str, groups: list[str], *, key: str, height: int = 430,
                custom: tuple | None = None, extra_levels=(), extra_lines=(), note_prefix: str = "") -> None:
    """Touch chart for one ticker: candles + EMA 9 / EMA 200 / volume / MACD, plus key levels."""
    bars = load_bars(t, tf, demo, custom)
    label = f"{custom[2]} · {lvl.fmt_date(custom[0])} – {lvl.fmt_date(custom[1])}" if tf == "Custom" and custom else tf
    intraday = is_intraday(tf, custom)
    if bars.empty:
        tvchart.render(bars, key=key, title=f"{t} · {label}")
        return
    lo, hi = float(bars["Low"].min()), float(bars["High"].max())
    span = max(hi - lo, hi * 0.002)
    raw: list[tuple] = []          # (price, title, color, style)
    notes: list[str] = []
    markers: list[dict] = []

    def add(y, text, color, dash):
        if lo - 0.6 * span <= y <= hi + 0.6 * span:     # skip levels far off the chart
            raw.append((float(y), text, color, dash))

    L = load_levels(t, demo) if intraday else None
    if L is not None:
        for g in groups:
            for k in LEVEL_GROUPS[g]:
                if k not in L["levels"]:
                    continue
                name, color, dash = LEVEL_STYLE[k]
                if k in ("ORH", "ORL") and not L["orb_locked"]:
                    name += " (forming)"
                if k in ("PMH", "PML") and not L["pm_locked"]:
                    name += " (live)"
                add(L["levels"][k], name, color, dash)
        sr_res, sr_sup = L["resistance"], L["support"]
        if "Opening range 9:30–10:00" in groups:
            first = bars.index[0]
            markers = [{"time": b["time"], "up": b["side"] == "up",
                        "text": f"OR break {'up' if b['side'] == 'up' else 'down'}"}
                       for b in L["breaks"] if b["time"] >= first]
            if L["breaks"]:
                notes.append("OR break: " + ", ".join(f"{b['side']} {b['time']:%H:%M}" for b in L["breaks"]))
            elif L["orb_locked"]:
                notes.append("OR: no break yet")
        lvls = L["levels"]
        if "PMH" in lvls:
            notes.insert(0, f"PM {lvls['PMH']:,.2f} / {lvls['PML']:,.2f}" + ("" if L["pm_locked"] else " (live)"))
        if "ORH" in lvls:
            notes.insert(1, f"OR {lvls['ORH']:,.2f} / {lvls['ORL']:,.2f}"
                         + (" (locked)" if L["orb_locked"] else " (forming)"))
    elif intraday:
        sr_res, sr_sup = [], []
    else:
        sr = load_long_sr(t, demo, tf in ("Weekly", "Max") or (custom is not None and custom[2] == "Weekly"))
        sr_res, sr_sup = lvl.nearest_sr(sr, float(bars["Close"].iloc[-1]), n=3)
    if "Support / resistance" in groups:
        for r in sr_res:
            add(r["price"], f"Resistance ({r['touches']}×)", DOWN, "dot")
        for x in sr_sup:
            add(x["price"], f"Support ({x['touches']}×)", UP, "dot")
        if sr_res:
            notes.append("R " + ", ".join(f"{r['price']:,.2f}" for r in sr_res))
        if sr_sup:
            notes.append("S " + ", ".join(f"{x['price']:,.2f}" for x in sr_sup))
    for lv_ in extra_levels:
        raw.append((float(lv_["price"]), lv_["title"], lv_.get("color", MUTED), lv_.get("style", "dash")))

    # Levels too close to read apart share one label (the others draw as plain lines).
    gap = 0.02 * (max([hi] + [r[0] for r in raw]) - min([lo] + [r[0] for r in raw]))
    levels, group = [], []
    for item in sorted(raw) + [None]:
        if item is not None and group and item[0] - group[-1][0] <= gap:
            group.append(item)
            continue
        if group:
            title = " · ".join(g[1] for g in reversed(group))
            for n, (y, _, color, dash) in enumerate(reversed(group)):
                levels.append({"price": y, "title": title if n == 0 else "", "color": color,
                               "style": dash, "label": n == 0})
        group = [item] if item is not None else []

    if candle_style == "Heikin Ashi":
        bars = bars.copy()
        bars[["Open", "High", "Low", "Close"]] = an.heikin_ashi(bars[["Open", "High", "Low", "Close"]])
        label += " · Heikin Ashi"
    tvchart.render(bars, key=key, title=f"{t} · {label}", height=height, intraday=intraday,
                   style=candle_style, levels=levels, markers=markers, lines=extra_lines,
                   note=" · ".join(([note_prefix] if note_prefix else []) + notes))


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
@st.cache_data(ttl=300, show_spinner=False)
def check_feed(key_tail: str) -> str:
    """Make one real Unusual Whales call so the badge reflects whether data actually flows."""
    try:
        return "ok" if len(uw.candles("SPY", "5m")) else "empty"
    except uw.UWError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"{e.__class__.__name__}: {e}"


key = uw.api_key()
if demo:
    feed = '<span class="dot off"></span>UW FEED: <b class="off">DEMO DATA</b>'
elif not key:
    feed = '<span class="dot off"></span>UW FEED: <b class="off">NO KEY</b>'
else:
    feed_state = check_feed(key[-4:])
    if feed_state == "ok":
        feed = '<span class="dot ok"></span>UW FEED: <b class="ok">LIVE</b>'
    else:
        feed = '<span class="dot off"></span>UW FEED: <b class="off">ERROR</b>'
        st.error(f"Unusual Whales isn't returning data: {feed_state}")
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
    <div class="pill">NY {now:%a %d/%b/%Y %H:%M} · {status}{(" · " + status_note) if status_note else ""}</div>
  </div>
</div>
""", unsafe_allow_html=True)

SECTIONS = ["🌅 Pre-Market", "🔎 Scanner", "📊 Charts & Levels", "🗂️ Sectors", "🧭 PM → Close", "📐 Setups & Patterns",
            "🐋 Options Flow", "🎯 Options & LEAPS", "📅 Earnings", "📈 Swing Odds",
            "⚡ Day Trading", "🧩 My Scripts"]


def _jump():
    """Remember which section was picked, then clear the menu so it's ready for the next jump."""
    st.session_state["jump_to"] = st.session_state.get("section_menu")
    st.session_state["section_menu"] = None


st.selectbox("Go to section", SECTIONS, index=None, key="section_menu", on_change=_jump,
             placeholder="☰  Jump to a section…", label_visibility="collapsed")
tabs = st.tabs(SECTIONS)
target = st.session_state.pop("jump_to", None)
if target in SECTIONS:
    # Streamlit can't switch tabs from Python, so a tiny script clicks the matching header tab.
    import streamlit.components.v1 as _components
    _components.html(f"""<script>
      const want = {SECTIONS.index(target)};
      const go = (n) => {{
        // first tab bar on the page = the header tabs (works for both old and new Streamlit tab markup)
        const bar = window.parent.document.querySelector('[role="tablist"]');
        const tabs = bar ? bar.querySelectorAll('[role="tab"]') : [];
        if (tabs.length > want) {{
          const t = tabs[want];
          t.dispatchEvent(new PointerEvent('pointerdown', {{bubbles: true, pointerType: 'mouse'}}));
          t.dispatchEvent(new PointerEvent('pointerup', {{bubbles: true, pointerType: 'mouse'}}));
          t.click();
          t.scrollIntoView({{block: "nearest", inline: "center"}});
        }} else if (n < 30) setTimeout(() => go(n + 1), 100);
      }};
      go(0);
    </script>""", height=0)

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

    if crypto_list:
        with st.expander(f"🪙 Crypto ({len(crypto_list)}) – ✅ Recognized by SEC and CFTC", expanded=False):
            st.caption("Digital commodities named in the SEC/CFTC release. Crypto trades 24/7, so there "
                       "is no pre-market or opening range – changes are vs. 24 hours, 7 and 30 days ago. "
                       "Edit this list in the CRYPTO section of `watchlist.txt`.")
            cx = load_crypto(tuple(crypto_list), demo)
            st.dataframe(cx.style.format({"Price": lambda v: "—" if pd.isna(v) else f"{v:,.6f}" if v < 0.01
                                          else f"{v:,.4f}" if v < 1 else f"{v:,.2f}", "24h %": "{:+.2f}%", "7d %": "{:+.2f}%",
                                          "30d %": "{:+.2f}%"}, na_rep="—")
                         .map(signed_color, subset=["24h %", "7d %", "30d %"]),
                         hide_index=True, width="stretch")

    @st.fragment(run_every=run_every)
    def watchlist_gaps():
        st.subheader("Watchlist gaps")
        with st.spinner("Loading pre-market data…"):
            pm = load_today(tuple(tickers), demo)
        cols_pm = [c for c in ("Ticker", "Prev close", "PM last", "Gap %", "Gap $", "PM high", "PM low",
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
                                 "Gap %": "{:+.2f}%", "Gap $": usd_signed}, na_rep="—")
                .map(signed_color, subset=[c for c in ["Gap %", "Gap $"] if c in pm.columns])
                .map(bias_color, subset=[c for c in ["PM bias"] if c in pm.columns]),
                hide_index=True, width="stretch", height=min(38 * (len(pm) + 1), 620))
            missing = pm[pm["Behavior"] == "no pre-market trades"]["Ticker"].tolist() if "Behavior" in pm else []
            if missing:
                st.info(f"No pre-market trades yet for: {', '.join(missing)}. Thinly traded ETFs "
                        "often don't print until ~8:00 ET – the row will fill once they trade.")
        with right:
            # same order as the table: biggest move (up or down) at the top
            g = (pm.dropna(subset=["Gap %"]).pipe(lambda d: d.reindex(d["Gap %"].abs().sort_values().index))
                 if "Gap %" in pm
                 else pd.DataFrame(columns=["Ticker", "Gap %"]))
            gap_usd = g["Gap $"] if "Gap $" in g else pd.Series(np.nan, index=g.index)
            fig = go.Figure(go.Bar(x=g["Gap %"], y=g["Ticker"], orientation="h",
                                   marker_color=[UP if v > 0 else DOWN for v in g["Gap %"]],
                                   customdata=[usd_signed(v) for v in gap_usd],
                                   text=[f"{p:+.2f}% · {usd_signed(v)}" for p, v in zip(g["Gap %"], gap_usd)],
                                   textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}: %{x:+.2f}% (%{customdata})<extra></extra>"))
            fig.update_layout(title="Gap vs prior close (% and $)", height=max(300, 26 * len(g)),
                              margin=dict(l=10, r=10, t=40, b=10), xaxis_ticksuffix="%")
            if len(g):   # room for the labels at the bar ends
                lo, hi = min(g["Gap %"].min(), 0), max(g["Gap %"].max(), 0)
                pad = (hi - lo or 1) * 0.45
                fig.update_xaxes(range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
            st.plotly_chart(fig, width="stretch", theme="streamlit")

        sess = pd.to_datetime(pd.Series(load_today(tuple(tickers), demo).get("Session date", pd.Series(dtype=object))),
                              errors="coerce").dropna()
        today_et = data.now_et().date()
        if len(sess):
            latest = sess.max().date()
            st.caption(f"Data for **{lvl.fmt_date(latest)}** · updated {data.now_et():%H:%M:%S} ET"
                       + (" · auto-refresh on" if live else " · turn on Auto-refresh in the sidebar for live updates"))
            if latest < today_et and data.now_et().weekday() < 5 and data.now_et().hour >= 4:
                st.warning(f"These numbers are from {lvl.fmt_date(latest)}, not today. Today's candles "
                           "haven't arrived yet – refresh in a minute, or check the UW FEED badge.")

    watchlist_gaps()

    st.subheader("Key levels – set automatically for every watchlist ticker")
    st.caption("Pre-market high/low lock at 9:30 · opening range (9:30–10:00) locks at 10:00 · "
               "Resistance/Support = nearest daily swing levels (× = how many times price turned there).")

    @st.fragment(run_every=run_every)
    def key_levels_table():
        rows = []
        for t in tickers:
            L = load_levels(t, demo)
            if L is None:
                rows.append({"Ticker": t, "OR status": "No data – check the symbol"})
                continue
            v = L["levels"]
            brk = ", ".join(f"{'Up' if b['side'] == 'up' else 'Down'} {b['time']:%H:%M}" for b in L["breaks"])
            rows.append({
                "Ticker": t, "Price": L["price"],
                "PM high": v.get("PMH"), "PM low": v.get("PML"),
                "OR high": v.get("ORH"), "OR low": v.get("ORL"),
                "OR status": ("—" if "ORH" not in v else "Forming" if not L["orb_locked"]
                              else (brk or "No break")),
                "Prev high": v.get("PDH"), "Prev low": v.get("PDL"), "Prev close": v.get("PDC"),
                "Resistance 1": L["resistance"][0]["price"] if L["resistance"] else None,
                "Resistance 2": L["resistance"][1]["price"] if len(L["resistance"]) > 1 else None,
                "Support 1": L["support"][0]["price"] if L["support"] else None,
                "Support 2": L["support"][1]["price"] if len(L["support"]) > 1 else None,
            })
        kl = pd.DataFrame(rows)
        num = [c for c in kl.columns if c not in ("Ticker", "OR status")]
        st.dataframe(kl.style.format({c: "{:,.2f}" for c in num}, na_rep="—"),
                     hide_index=True, width="stretch", height=min(38 * (len(kl) + 1), 640))

    key_levels_table()
    st.caption("Charts with these lines drawn in: **📊 Charts & Levels** tab.")

# --------------------------------------------------------------------------- #
# 0a. Scanner
# --------------------------------------------------------------------------- #
SCAN_HELP = {
    "Score": "How many of the 5 signals point the same way (gap, relative volume, PM range, opening range, "
             "options flow). Relative volume counts for both directions. 4–5 = strong agreement.",
    "Direction": "▲ = most signals bullish, ▼ = most bearish.",
    "Signals": "Which signals are on right now.",
    "Change %": "Price now vs yesterday's 4:00 PM close.",
    "Gap %": "Before 9:30: last pre-market price vs yesterday's close. After 9:30: today's open vs yesterday's close.",
    "Gap $": "The same gap in dollars per share.",
    "RVOL": "Relative volume – trading since 4:00 AM vs the usual amount by this time of day. 2.0x = twice normal.",
    "PM status": "Where price sits vs the pre-market high/low (4:00–9:29).",
    "ORB status": "Where price sits vs the 9:30–10:00 opening range. Fills in after 10:00.",
    "vs VWAP": "Above/below the volume-weighted average price (pre-market VWAP before 9:30).",
    "Calls %": "Share of unusual options premium in calls (latest session). 60%+ leans bullish, 40%- bearish.",
    "Gamma": "Positive = dealers damp moves (choppy, pinned). Negative = dealers add to moves (trending, bigger ranges).",
    "Call wall": "Strike with the most call gamma – often acts as resistance.",
    "Put wall": "Strike with the most put gamma – often acts as support.",
    "Gamma flip": "Estimated price where gamma switches from positive to negative.",
}

with tabs[1]:
    st.caption("Finds the tickers worth a look **right now**. Each ticker is checked for 5 signals – "
               "**gap**, **relative volume**, **pre-market range break**, **opening-range break** and "
               "**options flow** – and gets a **Score** for how many agree. Hover a column title for its meaning.")
    c1, c2 = st.columns([1.4, 2])
    scope = c1.radio("Scan", ["Watchlist", "Sector", "Tickers I type"], horizontal=True, key="scan_scope")
    universe = tickers
    if scope == "Sector":
        try:
            sp_all = load_sp500()
            sec_pick = c2.selectbox("Sector (S&P 500)", sorted(sp_all["Sector"].unique()), key="scan_sector")
            universe = sp_all.loc[sp_all["Sector"] == sec_pick, "Ticker"].tolist()
        except Exception as e:  # noqa: BLE001
            c2.warning(f"Couldn't load the sector list ({e.__class__.__name__}).")
            universe = []
    elif scope == "Tickers I type":
        universe = data.read_watchlist(c2.text_input("Tickers", "", placeholder="e.g. NVDA, AMD, SMCI",
                                                     key="scan_typed"))[0]
    filt = st.expander("⚙️ Filters – direction, minimum gap, volume, score, gamma levels")
    want_gamma = filt.toggle("Gamma levels", value=scope != "Sector", key=f"scan_gamma_{scope}",
                           help="Call wall, put wall, gamma flip and regime from Unusual Whales. Uses one "
                                "extra request per ticker – best left off for big sector scans.")
    f1, f2, f3, f4 = filt.columns(4)
    direction = f1.selectbox("Direction", ["Both", "Bullish only", "Bearish only"], key="scan_dir")
    min_gap = f2.number_input("Min gap %", 0.0, 50.0, 2.0, 0.5, key="scan_gap",
                              help="A gap at least this big (up or down) counts as a signal.")
    min_rvol = f3.number_input("Min relative volume", 0.5, 20.0, 2.0, 0.5, key="scan_rvol", format="%.1f",
                               help="Relative volume at least this high counts as a signal. 2.0 = twice normal.")
    min_score = f4.slider("Min score", 0, 5, 0, key="scan_score",
                          help="Hide tickers with fewer agreeing signals. 0 = show everything.")
    MIN_FLOW = 100_000

    if scope == "Watchlist":
        st.session_state["scan_for"] = tuple(universe)
    elif st.button(f"🔎 Scan {len(universe)} tickers", type="primary", disabled=not universe):
        st.session_state["scan_for"] = tuple(universe)

    @st.fragment(run_every=run_every)
    def scanner_table():
        tl = st.session_state.get("scan_for")
        if not tl or list(tl) != list(universe):
            st.info("Pick what to scan, then press **Scan**.")
            return
        has_uw = bool(uw.api_key()) and not demo
        flow = pd.DataFrame()
        if has_uw:
            try:
                flow = load_flow_summary(tuple(tl), MIN_FLOW)
            except Exception as e:  # noqa: BLE001
                st.caption(f"Options flow unavailable ({e}).")
        rows = []
        prog = st.progress(0.0, text="Scanning…") if len(tl) > 12 else None
        for i, t in enumerate(tl):
            gm = None
            if want_gamma and has_uw:
                try:
                    gm = load_gamma(t)
                except Exception:  # noqa: BLE001
                    gm = None
            rows.append(sc.scan_row(t, load_intraday(t, demo), flow.loc[t] if t in flow.index else None, gm,
                                    min_gap=min_gap, min_rvol=min_rvol, min_flow=MIN_FLOW))
            if prog:
                prog.progress((i + 1) / len(tl), text=f"Scanning {i + 1}/{len(tl)}: {t}")
        if prog:
            prog.empty()
        res = pd.DataFrame(rows)
        if "Gap %" not in res:
            st.warning("No data for these tickers yet.")
            return
        for c in ("Price", "Change %", "Gap %", "Gap $", "RVOL", "PM high", "PM low", "ORB high", "ORB low", "Calls %",
                  "Calls $", "Puts $", "Call wall", "Put wall", "Gamma flip"):
            if c in res:
                res[c] = pd.to_numeric(res[c], errors="coerce").astype(float)
        res = res.assign(_g=res["Gap %"].abs()).sort_values(["Score", "_g"], ascending=False,
                                                            na_position="last").drop(columns="_g")
        m = st.columns(5)
        m[0].metric("Scanned", len(res))
        m[1].metric("▲ Bullish (score 3+)", int(((res["Score"] >= 3) & res["Direction"].str.startswith("▲")).sum()))
        m[2].metric("▼ Bearish (score 3+)", int(((res["Score"] >= 3) & res["Direction"].str.startswith("▼")).sum()))
        m[3].metric(f"RVOL {min_rvol:g}x+", int((res["RVOL"] >= min_rvol).sum()))
        m[4].metric("Unusual flow", int((res.get("Flow alerts", pd.Series(dtype=float)) > 0).sum()))

        shown = res[res["Score"] >= min_score]
        if direction != "Both":
            shown = shown[shown["Direction"].str.startswith("▲" if direction.startswith("Bull") else "▼")]
        cols = [c for c in ("Ticker", "Score", "Direction", "Signals", "Price", "Change %", "Gap %", "Gap $", "RVOL",
                            "PM status", "PM high", "PM low", "ORB status", "ORB high", "ORB low", "vs VWAP",
                            "Calls %", "Calls $", "Puts $", "Gamma", "Call wall", "Put wall", "Gamma flip")
                if c in shown.columns]
        if not has_uw:
            cols = [c for c in cols if c not in ("Calls %", "Calls $", "Puts $", "Gamma", "Call wall",
                                                 "Put wall", "Gamma flip")]
        elif not want_gamma:
            cols = [c for c in cols if c not in ("Gamma", "Call wall", "Put wall", "Gamma flip")]
        money = {c: (lambda v: "—" if pd.isna(v) else f"${v / 1e6:,.1f}M" if v >= 1e6 else f"${v / 1e3:,.0f}K")
                 for c in ("Calls $", "Puts $")}
        price_cols = ("Price", "PM high", "PM low", "ORB high", "ORB low", "Call wall", "Put wall", "Gamma flip")
        styled = (shown[cols].style
                  .format({c: "{:,.2f}" for c in price_cols if c in cols}
                          | {"Change %": "{:+.2f}%", "Gap %": "{:+.2f}%", "RVOL": "{:.1f}x",
                             "Calls %": "{:.0f}%", "Gap $": usd_signed} | money, na_rep="—")
                  .map(signed_color, subset=[c for c in ("Change %", "Gap %", "Gap $") if c in cols])
                  .map(lambda v: f"color: {UP}" if str(v).startswith("▲") else
                       f"color: {DOWN}" if str(v).startswith("▼") else "", subset=["Direction"]))
        st.dataframe(styled, hide_index=True, width="stretch", height=min(38 * (len(shown) + 1), 720),
                     column_config={c: st.column_config.Column(help=h) for c, h in SCAN_HELP.items() if c in cols})
        if shown.empty:
            st.info("Nothing matches these filters – lower **Min score** or the gap / volume minimums.")
        st.caption(f"Updated {data.now_et():%H:%M:%S} ET · flow = unusual options alerts of "
                   f"${MIN_FLOW:,}+ from the latest session"
                   + ("" if live else " · turn on Auto-refresh in the sidebar for live updates"))
        st.download_button("Download this scan (CSV)", res.to_csv(index=False),
                           file_name=f"scan_{data.now_et():%Y-%m-%d_%H%M}.csv", key="scan_csv")

    scanner_table()

    with st.expander("How to read the scanner"):
        st.markdown(
            "- **Score 4–5** with ▲: gap up, heavy volume, holding above the pre-market high / opening range, "
            "and call-heavy flow – the strongest long setups.\n"
            "- **Score 4–5** with ▼: the same picture to the downside.\n"
            "- **High score + Negative gamma**: moves tend to run further (dealers chase price).\n"
            "- **High score + Positive gamma** near a **call wall** (longs) or **put wall** (shorts): "
            "moves often stall there – a natural profit target.\n"
            "- **Before 10:00** the opening-range signal can't fire yet, so the top possible score is 4.\n"
            "- The scanner finds candidates – confirm on the chart (📊 Charts & Levels) before trading. "
            "Not financial advice.")

# --------------------------------------------------------------------------- #
# 0b. Multi-ticker charts with levels
# --------------------------------------------------------------------------- #
with tabs[2]:
    picks = st.multiselect("Tickers", tickers, default=tickers[:4], max_selections=8, key="chart_picks")
    tf, custom = tf_picker("chart_tf", "5m")
    groups = st.multiselect("Lines", list(LEVEL_GROUPS), default=list(LEVEL_GROUPS), key="chart_groups")
    if not is_intraday(tf, custom):
        st.caption("Daily and longer charts show longer-term support & resistance. Pre-market and "
                   "opening-range lines appear on the 1-minute to 1-hour timeframes.")
    if not picks:
        st.info("Pick at least one ticker.")

    @st.fragment(run_every=run_every)
    def chart_grid():
        cols = st.columns(2)
        for i, t in enumerate(picks):
            with cols[i % 2]:
                price_chart(t, tf, groups, custom=custom, key=f"grid_{t}_{tf}")

    chart_grid()

# --------------------------------------------------------------------------- #
# 1. Sectors: every S&P 500 sector, pre-market scan + company lookup
# --------------------------------------------------------------------------- #
with tabs[3]:
    try:
        sp = load_sp500()
    except Exception as e:  # noqa: BLE001
        st.warning(f"Couldn't load the S&P 500 sector list ({e.__class__.__name__}). "
                   "Showing your watchlist as one group instead.")
        sp = pd.DataFrame({"Ticker": tickers, "Company": tickers, "Sector": "Watchlist",
                           "Sub-industry": "—"})
    sector_names = sorted(sp["Sector"].unique())
    c1, c2 = st.columns([2, 1])
    sec = c1.selectbox("Sector", sector_names + ["ALL sectors (slow – ~500 tickers)"], key="sec_pick")
    lookup = c2.text_input("Look up any ticker", "", placeholder="e.g. NVDA").strip().upper()

    if lookup:
        hit = sp[sp["Ticker"] == lookup]
        info = hit.iloc[0] if len(hit) else None
        st.markdown(f"**{lookup}** — {info['Company']} · {info['Sector']} · {info['Sub-industry']}"
                    if info is not None else f"**{lookup}** (not in the S&P 500 list)")
        one = load_today((lookup,), demo)
        cols1 = [c for c in ("Ticker", "Prev close", "PM last", "Gap %", "PM high", "PM low", "PM VWAP",
                             "PM bias", "PM volume", "Last PM print", "ORB high", "ORB low",
                             "ORB break", "Behavior") if c in one.columns]
        st.dataframe(one[cols1].style.format({"Gap %": "{:+.2f}%"}, precision=2, na_rep="—")
                     .map(bias_color, subset=[c for c in ["PM bias"] if c in cols1]),
                     hide_index=True, width="stretch")

    members = sp if sec.startswith("ALL") else sp[sp["Sector"] == sec]
    with st.expander(f"{len(members)} companies in {sec.split(' (')[0]}"):
        st.dataframe(members, hide_index=True, width="stretch", height=420)

    if st.button(f"🔎 Scan pre-market for {sec.split(' (')[0]}", type="primary"):
        st.session_state["sector_scanned"] = sec
    if st.session_state.get("sector_scanned") == sec:
        rows = []
        prog = st.progress(0.0, text="Scanning…")
        tl = members["Ticker"].tolist()
        for i, t in enumerate(tl):
            rows.append(load_today((t,), demo))
            prog.progress((i + 1) / len(tl), text=f"Scanning {i + 1}/{len(tl)}: {t}")
        prog.empty()
        res = pd.concat(rows, ignore_index=True).merge(members, on="Ticker", how="left")
        g = res.dropna(subset=["Gap %"]) if "Gap %" in res else res.iloc[:0]
        if g.empty:
            st.info("No pre-market prints yet for this group.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Gapping up", int((g["Gap %"] > 0).sum()))
            m2.metric("Gapping down", int((g["Gap %"] < 0).sum()))
            m3.metric("Average gap", f"{g['Gap %'].mean():+.2f}%")
            m4.metric("Bullish PM bias", f"{(g['PM bias'] == 'BULLISH').mean() * 100:.0f}%")
            show = [c for c in ("Ticker", "Company", "Sub-industry", "PM last", "Gap %", "PM high",
                                "PM low", "PM bias", "PM volume") if c in g.columns]
            fmt = {"Gap %": "{:+.2f}%", "PM last": "{:,.2f}", "PM high": "{:,.2f}", "PM low": "{:,.2f}",
                   "PM volume": "{:,.0f}"}
            left, right = st.columns(2)
            with left:
                st.markdown("**Top gainers**")
                st.dataframe(g.nlargest(10, "Gap %")[show].style.format(fmt, na_rep="—")
                             .map(signed_color, subset=["Gap %"]), hide_index=True, width="stretch")
            with right:
                st.markdown("**Top losers**")
                st.dataframe(g.nsmallest(10, "Gap %")[show].style.format(fmt, na_rep="—")
                             .map(signed_color, subset=["Gap %"]), hide_index=True, width="stretch")
            group = "Sector" if sec.startswith("ALL") else "Sub-industry"
            by = g.groupby(group)["Gap %"].mean().sort_values()
            fig = go.Figure(go.Bar(x=by.values, y=by.index, orientation="h",
                                   marker_color=[UP if v > 0 else DOWN for v in by.values],
                                   hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
            fig.update_layout(title=f"Average pre-market gap by {group.lower()}",
                              height=max(300, 24 * len(by)), margin=dict(l=10, r=10, t=40, b=10),
                              xaxis_ticksuffix="%")
            st.plotly_chart(fig, width="stretch", theme="streamlit")
            st.markdown("**Every ticker**")
            st.dataframe(res[[c for c in show + ["Behavior"] if c in res.columns]]
                         .sort_values("Gap %", ascending=False, na_position="last")
                         .style.format(fmt, na_rep="—").map(signed_color, subset=["Gap %"])
                         .map(bias_color, subset=[c for c in ["PM bias"] if c in res.columns]),
                         hide_index=True, width="stretch", height=520)
            st.download_button("Download this scan (CSV)", res.to_csv(index=False),
                               file_name=f"premarket_{sec.split(' ')[0]}_{data.now_et():%Y-%m-%d}.csv")

# --------------------------------------------------------------------------- #
# 1a. Pre-market setup vs what happened by the afternoon
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.caption("Your pre-market rules: **PM bias** = last pre-market price vs pre-market VWAP. "
               "**Opening range** = 9:30–10:00. **Behavior** compares the price after 10:00 with "
               "the opening range: **Breakout ▲** = above the 10:00 high, **Breakdown ▼** = below the 10:00 low; "
               "*with* / *against* says whether that matched the pre-market bias. **Afternoon** = 1:00–4:00 PM.")

    @st.fragment(run_every=run_every)
    def pm_to_close():
        with st.spinner("Loading today's sessions…"):
            td = load_today(tuple(tickers), demo)
        cols = [c for c in ("Ticker", "PM bias", "Gap %", "PM high", "PM low", "PM VWAP",
                            "ORB high", "ORB low", "ORB range %", "ORB break", "Ext above ORB (x)",
                            "Ext below ORB (x)", "Price", "Behavior", "Took PM high", "Took PM low",
                            "Morning %", "Afternoon %", "Afternoon followed PM?", "Close vs PM VWAP")
                if c in td.columns]
        st.subheader("Today")
        st.dataframe(
            td[cols].style.format({c: "{:,.2f}" for c in ("PM high", "PM low", "PM VWAP", "ORB high",
                                                           "ORB low", "Price") if c in cols}
                                  | {c: "{:+.2f}%" for c in ("Gap %", "Morning %", "Afternoon %")
                                     if c in cols}
                                  | {"ORB range %": "{:.2f}%", "Ext above ORB (x)": "{:.1f}x",
                                     "Ext below ORB (x)": "{:.1f}x"}, na_rep="—")
            .map(bias_color, subset=["PM bias"] if "PM bias" in cols else [])
            .map(signed_color, subset=[c for c in ("Gap %", "Morning %", "Afternoon %") if c in cols]),
            hide_index=True, width="stretch", height=min(38 * (len(td) + 1), 700))
        if live:
            st.caption(f"Live · updated {data.now_et():%H:%M:%S} ET")

    pm_to_close()

    st.subheader("History – how each ticker's pre-market setup usually plays out")
    st.caption("Last ~60 sessions of 5-minute data. Continued = a breakout in the same direction as the "
               "pre-market bias (bullish PM → above the opening range, bearish PM → below).")
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
        show = [c for c in ("Date", "PM bias", "Gap %", "ORB break", "Behavior", "Took PM high", "Took PM low",
                            "Morning %", "Afternoon %", "Afternoon followed PM?") if c in h.columns]
        st.dataframe(h[show].iloc[::-1].assign(Date=lambda x: x["Date"].map(lvl.fmt_date)).style.format({"Gap %": "{:+.2f}%", "Morning %": "{:+.2f}%",
                                                      "Afternoon %": "{:+.2f}%"}, na_rep="—")
                     .map(bias_color, subset=["PM bias"]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# 1b. Chart patterns / setups
# --------------------------------------------------------------------------- #
with tabs[5]:
    c1, c2, c3 = st.columns([1, 1, 2])
    tf = c1.radio("Scan on", ["Daily", "5-minute"], horizontal=True,
                  help="Which candles the pattern scanner looks at. Daily = swing setups. "
                       "5-minute = intraday setups (regular hours, last 60 days).")
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
    ctf, ccustom = tf_picker("pat_tf", "Daily" if tf == "Daily" else "5m")
    mine = [f for f in found if f["Ticker"] == pk]
    pat_lines = []
    # Pattern lines belong to the candles they were found on, so only draw them on that timeframe.
    if (tf == "Daily" and ctf == "Daily") or (tf == "5-minute" and ctf == "5m"):
        src = data.daily_history(pk, demo=demo).tail(150) if tf == "Daily" else load_bars(pk, ctf, demo, ccustom)
        if tf != "Daily":
            mm = src.index.hour * 60 + src.index.minute
            src = src[(mm >= 570) & (mm < 960)]
        for f in mine:
            for x0, y0, x1, y1, lbl in pt.pattern_lines(src, f["Pattern"]):
                pat_lines.append({"points": [(x0, y0), (x1, y1)], "style": "dash",
                                  "color": "#38bdf8" if lbl == "upper" else "#fb923c",
                                  "title": f"{f['Pattern']} ({lbl})"})
    elif mine:
        st.caption(f"Pattern lines are drawn on the {'Daily' if tf == 'Daily' else '5m'} timeframe "
                   "(the candles the scan used).")
    price_chart(pk, ctf, ["Support / resistance"], key=f"pat_{pk}_{ctf}", custom=ccustom,
                extra_lines=pat_lines)
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
with tabs[6]:
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
                px = last_price(ot)
                c3.metric("Price", "—" if pd.isna(px) else f"{px:,.2f}")
                st.markdown("**Today's options volume & premium**")
                ov = uw_panel("options_volume", ticker=ot)
                if ov is not None and len(ov):
                    st.dataframe(ov.head(5), hide_index=True, width="stretch")
                st.markdown(f"**Flow alerts – {ot}**")
                fa = uw_panel("flow_alerts", {"ticker_symbol": ot, "limit": 100})
                if fa is not None and len(fa):
                    st.dataframe(pretty_dates(fa), hide_index=True, width="stretch", height=380)
                else:
                    st.write("No flow alerts for this ticker today.")
                st.markdown(f"**Recent trades – {ot}**")
                fr = uw_panel("flow_ticker", ticker=ot)
                if fr is not None and len(fr):
                    st.dataframe(pretty_dates(fr.head(200)), hide_index=True, width="stretch", height=380)

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
                    agg = (df.assign(side=df[kcol].astype(str).str.lower(),
                                     **{pcol: pd.to_numeric(df[pcol], errors="coerce").fillna(0)})
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
                st.dataframe(pretty_dates(df[lead + [c for c in df.columns if c not in lead]]),
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
                    st.plotly_chart(style_fig(fig, 420, intraday=True), width="stretch", theme="streamlit")
                st.dataframe(pretty_dates(df.tail(50)), hide_index=True, width="stretch")

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
                st.dataframe(pretty_dates(df), hide_index=True, width="stretch", height=520)

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
with tabs[8]:
    st.subheader("Earnings this week")
    today_d = data.now_et().date()
    wk = st.radio("Week", ["This week", "Next week"], horizontal=True, key="earn_week")
    monday = today_d - dt.timedelta(days=today_d.weekday()) + dt.timedelta(weeks=wk == "Next week")
    friday = monday + dt.timedelta(days=4)
    st.caption(f"{monday:%a %d/%b/%Y} – {friday:%a %d/%b/%Y}. **BMO** = reports before the 9:30 open "
               "(trade the reaction that morning). **AMC** = reports after the 4:00 close "
               "(reaction is the next morning's pre-market).")
    ew_tabs = st.tabs(["Your watchlist", "All companies (Unusual Whales)"])
    with ew_tabs[0]:
        with st.spinner("Loading earnings dates…"):
            ew = load_earnings_week(tuple(tickers), monday, friday, demo)
        if ew.empty:
            st.info("None of your watchlist tickers report this week.")
        else:
            st.dataframe(ew.assign(Date=ew["Date"].map(lvl.fmt_date))
                         .style.format({"EPS est": "{:.2f}", "EPS actual": "{:.2f}"}, na_rep="—"),
                         hide_index=True, width="stretch")
    with ew_tabs[1]:
        if not uw.api_key() or demo:
            st.info("Needs your Unusual Whales key (and Demo data off).")
        else:
            frames = []
            for i in range(5):
                d = monday + dt.timedelta(days=i)
                for name, when in (("earnings_premarket", "Before open (BMO)"),
                                   ("earnings_afterhours", "After close (AMC)")):
                    df = uw_panel(name, {"date": d.isoformat()})
                    if df is not None and len(df):
                        frames.append(df.assign(Date=d, Day=f"{d:%a}", When=when))
            if frames:
                allw = pd.concat(frames, ignore_index=True)
                tcol = first_col(allw, "symbol", "ticker", "ticker_symbol")
                lead = [c for c in ("Date", "Day", "When", tcol, "full_name", "report_time",
                                    "expected_move_perc", "street_mean_est") if c and c in allw.columns]
                q = st.text_input("Filter (ticker)", "", key="earn_filter").upper()
                if q and tcol:
                    allw = allw[allw[tcol].astype(str).str.contains(q)]
                st.dataframe(pretty_dates(allw[lead + [c for c in allw.columns if c not in lead]]),
                             hide_index=True, width="stretch", height=520)
                st.caption("If every day shows the same list, Unusual Whales ignored the date "
                           "filter – tell Claude and it will switch to their per-ticker endpoint.")

    st.divider()
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
        st.dataframe(upcoming.assign(Date=upcoming["Date"].map(lvl.fmt_date))
                     .style.format({"EPS est": "{:.2f}", "Revenue est": "{:,.0f}"},
                                           na_rep="—"), hide_index=True, width="stretch")

        et = st.selectbox("Implied vs historical move", upcoming["Ticker"].tolist())
        edate = upcoming.set_index("Ticker").loc[et, "Date"]
        spot = last_price(et)
        exps = [e for e in load_expirations(et, demo) if pd.Timestamp(e).date() >= edate]
        daily = load_daily(et, demo)
        hist = an.earnings_reactions(daily, data.past_earnings_dates(et, demo=demo))
        c1, c2, c3 = st.columns(3)
        imp = np.nan
        if exps and pd.notna(spot):
            imp = an.implied_move(load_chain(et, exps[0], spot, demo), spot)
            c1.metric(f"Implied move (exp {lvl.fmt_date(exps[0])})", f"±{imp:.1f}%" if pd.notna(imp) else "—")
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
with tabs[9]:
    st.caption(f"Setups: trend pullback, breakout, oversold bounce · stop {stop_atr}×ATR · "
               f"target {target_atr}×ATR · max {max_days} days.")
    with st.spinner("Scanning setups (backtesting each ticker)…"):
        scans = load_swing(tuple(tickers), demo, stop_atr, target_atr, max_days)
    if not scans:
        st.info("No swing data for your watchlist – check the symbols, or turn on Demo data to preview.")
    else:
        sdf = pd.DataFrame([{k: v for k, v in s.items() if k != "_stats"} for s in scans])
        active = sdf[sdf["Setups today"] != "—"].copy()
        strong = (active["Hist win rate"] >= HIGH_PROB) & (active["Hist signals"] >= MIN_SIGNALS)
        hi_df = active[strong].sort_values("Hist win rate", ascending=False)
        lo_df = active[~strong].sort_values("Hist win rate", ascending=False, na_position="last")
        cols = ["Ticker", "Setups today", "Hist win rate", "Hist signals", "Price", "Trend", "RSI",
                "Entry", "Stop", "Target", "ATR %"]
        fmt = {"Price": "{:,.2f}", "RSI": "{:.0f}", "ATR %": "{:.1f}%", "Entry": "{:,.2f}",
               "Stop": "{:,.2f}", "Target": "{:,.2f}", "Hist win rate": "{:.0%}", "Hist signals": "{:.0f}"}
        ren = {"Hist win rate": "Win probability", "Hist signals": "Times seen"}
        fmt_r = {ren.get(k, k): v for k, v in fmt.items()}

        st.subheader(f"✅ High-probability setups (≥ {HIGH_PROB:.0%}) — {len(hi_df)}")
        if hi_df.empty:
            st.info(f"No setup today has won at least {HIGH_PROB:.0%} of the time (with {MIN_SIGNALS}+ past "
                    "occurrences). Setups below that are listed next with their probability.")
        else:
            st.dataframe(hi_df[cols].rename(columns=ren).style.format(fmt_r, na_rep="—"),
                         hide_index=True, width="stretch")

        st.subheader(f"Below {HIGH_PROB:.0%} — {len(lo_df)}")
        if lo_df.empty:
            st.caption("None.")
        else:
            lo_show = lo_df[cols].rename(columns=ren)
            lo_show.insert(3, "Note", ["small sample" if n < MIN_SIGNALS else ""
                                       for n in lo_df["Hist signals"]])
            st.dataframe(lo_show.style.format(fmt_r, na_rep="—"), hide_index=True, width="stretch")

        quiet = sdf[sdf["Setups today"] == "—"]["Ticker"].tolist()
        if quiet:
            st.caption(f"No setup today: {', '.join(quiet)}")
        st.caption(f"**Win probability** = of the past times this setup appeared on this ticker (~5 years of "
                   f"daily candles), how often price reached the target ({target_atr}×ATR) before the stop "
                   f"({stop_atr}×ATR) within {max_days} days. Change stop/target in the sidebar: a closer "
                   "target raises the win rate but lowers the payoff. **Times seen** = how many past "
                   f"occurrences; under {MIN_SIGNALS} is a small sample. Past results don't guarantee future ones.")

        sp = st.selectbox("Chart & setup history", [x["Ticker"] for x in scans], key="swing_pick")
        sc = next(s for s in scans if s["Ticker"] == sp)
        st.dataframe(pd.DataFrame(sc["_stats"]).T.rename(columns={"n": "Signals", "win_rate": "Win rate",
                                                                  "avg_ret": "Avg return"})
                     .style.format({"Win rate": "{:.0%}", "Avg return": "{:+.2%}", "Signals": "{:.0f}"},
                                   na_rep="—"), width="stretch")
        stf, scustom = tf_picker("swing_tf", "Daily")
        price_chart(sp, stf, ["Support / resistance"], key=f"swing_{sp}_{stf}", custom=scustom,
                    extra_levels=[{"price": sc["Stop"], "title": "Stop", "color": DOWN, "style": "dash"},
                                  {"price": sc["Target"], "title": "Target", "color": UP, "style": "dash"}])
        st.caption("Stop and target come from the daily setup scan above.")

# --------------------------------------------------------------------------- #
# 4. LEAPS
# --------------------------------------------------------------------------- #
with tabs[7]:
    c1, c2, c3, c4 = st.columns(4)
    lt = c1.selectbox("Ticker", tickers, key="leaps_pick")
    kind = c2.radio("Type", ["call", "put"], horizontal=True)
    min_dte = c3.number_input("Min days to expiry", 180, 1000, 365, 30)
    vol_src = c4.radio("Probability vol", ["Implied", "Historical (1y)"], horizontal=True,
                       help="Implied = market's view. Historical = what the stock has actually done.")
    spot = last_price(lt)
    daily = load_daily(lt, demo)
    if daily.empty or pd.isna(spot):
        no_data(lt)
    else:
        hv = an.realized_vol(daily["Close"], 252) if vol_src.startswith("Hist") else None
        exps = [e for e in load_expirations(lt, demo)
                if (pd.Timestamp(e).date() - data.now_et().date()).days >= min_dte]
        if not exps:
            st.warning(f"No expirations ≥ {min_dte} days for {lt}.")
        else:
            chain = pd.concat([load_chain(lt, e, spot, demo) for e in exps], ignore_index=True)
            tbl = an.leaps_table(chain, spot, rate, kind, hv)
            if tbl.empty or "Delta" not in tbl:
                st.info(f"No usable option quotes for {lt} right now.")
                tbl = pd.DataFrame(columns=["Fit score", "Expiry", "Delta"])
            dmin, dmax = st.slider("Delta range (abs)", 0.05, 0.99, (0.55, 0.95))
            tbl = tbl[tbl["Delta"].abs().between(dmin, dmax)]
            st.metric(f"{lt} spot", f"{spot:,.2f}")
            st.dataframe(
                tbl.assign(Expiry=tbl["Expiry"].map(lvl.fmt_date)).style.format({"Strike": "{:,.2f}", "Mid": "{:,.2f}", "IV %": "{:.0f}",
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
with tabs[10]:
    with st.expander("ℹ️ How to use this page"):
        st.markdown(
            "1. **Pick a ticker and a timeframe** (1m–5m for the open, 15m–1h for the trend of the day).\n"
            "2. **Read the four numbers**: is the stock moving more than normal (Relative volume), how far it "
            "gapped, and how much of a normal day's move is already used up (Range used).\n"
            "3. **Watch price against the lines on the chart.** The table below lists every line, its price, "
            "how far it is from the current price, and what it usually means.\n"
            "4. **Opening-range breakout:** after 10:00, a move above the *Opening range high* is bullish, "
            "below the *Opening range low* is bearish. Pre-market and prior-day levels are the next targets.\n"
            "5. **Chart colors:** purple = VWAP (above it = buyers in control today, below = sellers) · "
            "blue = EMA 9 (short-term trend) · white = EMA 200 (long-term trend) · bars at the bottom = "
            "volume · lower panel = MACD: red line crossing above yellow = momentum turning up, below = down.\n"
            "6. **Touch:** pinch to zoom the candles, drag sideways to move, swipe up/down to scroll the page. "
            "**Reset view** jumps back to the latest candles.")
    c1, c2 = st.columns([1, 3])
    dtk = c1.selectbox("Ticker", tickers, key="dt_pick")
    with c2:
        dtf, dcustom = tf_picker("dt_tf", "5m")
    intra = load_intraday(dtk, demo, interval="5m", days=5)
    if intra.empty or load_daily(dtk, demo).empty:
        no_data(dtk)
    else:
        info = an.day_levels(intra, load_daily(dtk, demo))
        lv, stx = info["levels"], info["stats"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ATR (14-day)", f"{stx['ATR']:.2f}",
                  help="**Average True Range** – how much this stock typically moves in one day (high to low, "
                       "including overnight gaps), averaged over 14 days.\n\nUse it to size stops and targets: "
                       "with ATR 3.00, a 1.00 move is about a third of a normal day. Higher ATR = wilder stock.")
        m2.metric("Gap", f"{stx['Gap %']:+.2f}%" if "Gap %" in stx else "—",
                  help="How far the **pre-market price** is from **yesterday's close**, in %.\n\n"
                       "Over ±2% usually means news. Gaps that hold above/below the pre-market high/low tend to "
                       "keep going; weak gaps often drift back toward yesterday's close ('gap fill').")
        rv = stx.get("RVOL")
        m3.metric("Relative volume", f"{rv:.2f}x" if rv is not None and pd.notna(rv) else "—",
                  help="**Today's volume so far ÷ the normal volume by this time of day.**\n\n"
                       "1.0x = normal · 2.0x = twice normal (strong interest – moves tend to follow through) · "
                       "under 1x = quiet (breakouts fail more often). Shows '—' before the open.")
        used = stx.get("ATR used %")
        m4.metric("Range used today", f"{used:.0f}%" if used is not None else "—",
                  help="**Today's high-to-low range as a % of ATR** – how much of a normal day's move has "
                       "already happened.\n\nUnder ~50% = room left to run · around 100% = a full normal day "
                       "already · over 100% = extended, new highs/lows get harder. Shows '—' before the open.")

        bars = load_bars(dtk, dtf, demo, dcustom)
        vw_now, vw_line = None, []
        if is_intraday(dtf, dcustom) and len(bars):
            # VWAP restarts each session (regular hours only); a gap between sessions, not a line
            mm = bars.index.hour * 60 + bars.index.minute
            rth = bars[(mm >= 570) & (mm < 960)]
            if len(rth):
                parts = [an.vwap(g) for _, g in rth.groupby(rth.index.date)]
                vwap_all = pd.concat(parts).reindex(bars.index)       # NaN outside regular hours = gap
                vw_line = [{"points": list(zip(bars.index, vwap_all.values)), "color": "#a855f7",
                            "style": "solid", "title": "VWAP"}]
                if rth.index[-1].date() == data.now_et().date():
                    vw_now = float(parts[-1].iloc[-1])
        price_chart(dtk, dtf, list(LEVEL_GROUPS), key=f"dt_{dtk}_{dtf}", height=520, custom=dcustom,
                    extra_lines=vw_line)
        st.caption("Purple = **VWAP** (today's volume-weighted average price). Blue = EMA 9, white = EMA 200, "
                   "bars at the bottom = volume, lower panel = MACD (red) and signal (yellow).")

        st.subheader("Levels – what each line means")
        price = last_price(dtk)
        rows = []
        for k, (name, meaning) in LEVEL_INFO.items():
            if k in lv:
                rows.append({"Level": name, "Price": lv[k], "Meaning": meaning})
        if vw_now is not None:
            rows.append({"Level": "VWAP (today)", "Price": vw_now,
                         "Meaning": "Volume-weighted average price. Above = buyers in control; below = sellers."})
        lt_df = pd.DataFrame(rows).sort_values("Price", ascending=False)
        if pd.notna(price):
            lt_df.insert(2, "Distance", [f"{(p / price - 1) * 100:+.2f}%" for p in lt_df["Price"]])
            lt_df.insert(3, "Where", ["above price ▲" if p > price else "below price ▼" for p in lt_df["Price"]])
            st.caption(f"Current price: **{price:,.2f}**. Levels are sorted high → low.")
        # st.table wraps long text, so every "Meaning" is readable in full
        st.table(lt_df.set_index("Level").style.format({"Price": "{:,.2f}"}))

# --------------------------------------------------------------------------- #
# 6. Plugins: your own scripts
# --------------------------------------------------------------------------- #
with tabs[11]:
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

# Data check: name the tickers that returned nothing, and nothing else.
_today = load_today(tuple(tickers), demo)
_missing = (_today.loc[_today["Behavior"].astype(str).str.startswith("No data"), "Ticker"].tolist()
            if "Behavior" in _today else [])
_coins = load_crypto(tuple(crypto_list), demo) if crypto_list else pd.DataFrame()
_missing_coins = _coins.loc[_coins["Price"].isna(), "Coin"].tolist() if len(_coins) else []
if _missing or _missing_coins:
    with st.sidebar:
        st.subheader("Data check")
        if _missing:
            st.warning(f"No data found for **{', '.join(_missing)}**. Check the spelling, or remove "
                       "it from your watchlist.")
        if _missing_coins:
            st.warning(f"No price for crypto **{', '.join(_missing_coins)}** right now.")
