"""Wiggins Investments LLC — Options Desk.   Run:  streamlit run app.py"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from bt import ai, charts, config, engine, store
from bt import metrics as mx
from bt import signals as sig
from bt.sources import get_source

COMPANY, OPERATOR = "Wiggins Investments LLC", "Sa'Rah El"
AUM = "$5 Million"            # Assets Under Management – edit here (same as dashboard)
VALUATION = "$50 Million"     # Market cap – edit here
F = config.fmt_date

st.set_page_config(page_title=f"{COMPANY} — Options Desk", page_icon="🧪", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.6rem; }
.hdr { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;
       background:linear-gradient(135deg, #1e3a8a 0%, #172554 100%); border:1px solid #2563eb;
       border-radius:14px; padding:14px 18px; margin-bottom:10px; box-shadow:0 6px 24px rgba(37,99,235,.25); }
.brand { display:flex; align-items:center; gap:12px; }
.logo { background:#2563eb; border-radius:10px; padding:6px 9px; font-size:20px; box-shadow:0 4px 14px rgba(59,130,246,.45); }
.name { font-weight:800; letter-spacing:.04em; color:#fff; font-size:18px; }
.badge { font: 700 10px ui-monospace, monospace; color:#93c5fd; background:rgba(30,58,138,.8);
         border:1px solid #3b82f6; border-radius:999px; padding:2px 8px; margin-left:6px; vertical-align:middle; }
.sub { color:#cbd5e1; font-size:12px; }
.kpi { color:#fff; font-weight:600; }
.status { display:flex; gap:8px; flex-wrap:wrap; }
.hdr .pill { background:#0f172a; border:1px solid rgba(96,165,250,.45); padding:5px 10px; margin:0; color:#e2e8f0; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.dot.ok { background:#10b981; animation:pulse 1.6s infinite; } .dot.off { background:#f59e0b; }
b.ok { color:#34d399; } b.off { color:#fbbf24; }
@keyframes pulse { 50% { opacity:.35; } }
.pill { display:inline-block; background:#0f172a; border:1px solid #1f2937; border-radius:8px; padding:4px 9px;
        font: 12px ui-monospace, monospace; color:#cbd5e1; margin:2px 4px 2px 0; }
.ok { color:#34d399; } .off { color:#fbbf24; }
.warn { background:#422006; border:1px solid #92400e; color:#fde68a; border-radius:10px; padding:8px 12px; font-size:13px; }
[data-testid="stMetricLabel"] p { color:#94a3b8 !important; }
[data-testid="stMetricValue"] { color:#f1f5f9 !important; }
[data-testid="stMetric"] { background:#111827; border:1px solid #1f2937; border-radius:12px; padding:10px 14px; }
[data-testid="stDataFrame"] { border:1px solid #1f2937; border-radius:12px; }
@media (max-width: 640px) {
  .block-container { padding-left:.6rem; padding-right:.6rem; }
  .name { font-size:14px; } .badge { display:none; }
  [data-testid="stMetricValue"] { font-size:1.25rem; }
  [data-testid="stMetric"] { padding:6px 10px; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:8px; }
  [data-testid="stColumn"] { min-width:calc(50% - 8px) !important; flex:1 1 calc(50% - 8px) !important; }
}
</style>
""", unsafe_allow_html=True)

keys = config.key_status()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    ready = {"Alpha Vantage": keys["ALPHAVANTAGE_API_KEY"],
             "ORATS + FMP": keys["ORATS_API_KEY"] and keys["FMP_API_KEY"],
             "Demo (synthetic)": True}
    choices = list(ready)
    default = next(c for c in choices if ready[c])
    src_name = st.radio("Data source", choices, index=choices.index(default),
                        help="Demo = random synthetic data, only for checking the app works.")
    live_ready = ready[src_name]
    if not live_ready:
        need = "ALPHAVANTAGE_API_KEY" if src_name == "Alpha Vantage" else "ORATS_API_KEY and FMP_API_KEY"
        st.error(f"{src_name} needs {need} in Secrets.")
    st.caption("Keys found in Secrets:")
    st.markdown("".join(
        f"<span class='pill'><b class='{'ok' if v else 'off'}'>{'●' if v else '○'}</b> {k.replace('_API_KEY','')}</span>"
        for k, v in keys.items()), unsafe_allow_html=True)
    s = store.stats()
    st.caption(f"Cache: {s['backend']} · {s['entries']:,} saved downloads")
    st.caption("Alpha Vantage covers option chains, prices, earnings and the 30-min opening range. "
               "Live data (Schwab, Unusual Whales) stays in your Options Desk.")

source = get_source(src_name)

try:
    from zoneinfo import ZoneInfo
    now_ny = dt.datetime.now(ZoneInfo("America/New_York"))
except Exception:
    now_ny = dt.datetime.utcnow()
if source.synthetic:
    feed = '<span class="dot off"></span>DATA: <b class="off">DEMO</b>'
elif not live_ready:
    feed = '<span class="dot off"></span>DATA: <b class="off">NO KEY</b>'
else:
    feed = f'<span class="dot ok"></span>{src_name.upper()}: <b class="ok">LIVE</b>'
st.markdown(f"""
<div class="hdr">
  <div class="brand">
    <div class="logo">📈</div>
    <div>
      <div class="name">{COMPANY.upper()} <span class="badge">OPTIONS DESK</span></div>
      <div class="sub">Operator: {OPERATOR}</div>
      <div class="sub">Assets Under Management <b class="kpi">{AUM}</b> · Market Cap <b class="kpi">{VALUATION}</b></div>
    </div>
  </div>
  <div class="status">
    <div class="pill">{feed}</div>
    <div class="pill">NY {now_ny:%a %d/%b/%Y %H:%M}</div>
  </div>
</div>
""", unsafe_allow_html=True)
if source.synthetic:
    st.markdown("<div class='warn'>⚠️ DEMO DATA – random synthetic prices. Use only to check the app works. "
                "Pick <b>Alpha Vantage</b> under Data source in the sidebar for real results.</div>", unsafe_allow_html=True)

PRESETS = {
    "Swing (45–60 DTE)": dict(dte=(45, 60), delta=(0.60, 0.70), pt=50, sl=40, ts=21, hold=0,
                              rev=sig.REVERSAL_RULES[1], spread=10.0, alloc=5.0, earn="Skip"),
    "LEAPS (1–2 years)": dict(dte=(365, 730), delta=(0.70, 0.85), pt=100, sl=35, ts=120, hold=0,
                              rev=sig.REVERSAL_RULES[2], spread=8.0, alloc=10.0, earn="Hold through"),
}

tab_setup, tab_res, tab_trades, tab_ai = st.tabs(["⚙️ Setup", "📊 Results", "📜 Trades", "🤖 AI Analysis"])

# --------------------------------------------------------------------------- setup
with tab_setup:
    preset = st.radio("Strategy", list(PRESETS), horizontal=True)
    P = PRESETS[preset]
    k = lambda name: f"{preset}:{name}"          # new defaults when strategy changes

    with st.form("setup"):
        st.subheader("Universe & dates")
        c1, c2, c3 = st.columns([2, 1, 1])
        tickers_txt = c1.text_input("Tickers (comma separated)", "SPY, QQQ, AAPL, MSFT, NVDA", key=k("tick"))
        start = c2.date_input("Start", dt.date(2015, 1, 1), format="DD/MM/YYYY", key=k("start"),
                              min_value=dt.date(2007, 1, 1))
        end = c3.date_input("End", dt.date.today() - dt.timedelta(days=1), format="DD/MM/YYYY", key=k("end"))
        direction = st.radio("Direction", ["Calls & Puts", "Calls only", "Puts only"], horizontal=True, key=k("dir"))

        st.subheader("Contract selection")
        c1, c2, c3, c4 = st.columns(4)
        dte = c1.slider("DTE at entry", 7, 900, P["dte"], key=k("dte"))
        delta = c2.slider("Delta (absolute)", 0.20, 0.95, P["delta"], 0.01, key=k("delta"))
        spread = c3.number_input("Max bid/ask spread %", 1.0, 50.0, P["spread"], 0.5, key=k("spread"),
                                 help="(ask − bid) ÷ mid. Skips illiquid contracts.")
        min_oi = c4.number_input("Min open interest", 0, 100_000, 100, 50, key=k("oi"))

        st.subheader("Entry rules (all ticked rules must be true on the signal day's close)")
        c1, c2, c3, c4 = st.columns(4)
        r_pull = c1.checkbox("Trend pullback", True, key=k("pull"),
                             help="Calls: price > EMA200, EMA50 > EMA200, yesterday closed below EMA9, today back above. Puts mirror.")
        r_brk = c1.checkbox("Breakout", False, key=k("brk"), help="Close above prior N-day high (puts: below N-day low).")
        brk_len = c1.number_input("Breakout days", 5, 120, 20, key=k("brklen"))
        r_ema = c2.checkbox("EMA 9 / EMA 200", False, key=k("ema"), help="Calls: EMA9 > EMA200 and close > EMA9.")
        r_macd = c2.checkbox("MACD 12/26/9 cross", False, key=k("macd"))
        macd_lb = c2.number_input("Cross within last N days", 1, 10, 3, key=k("macdlb"))
        r_ivr = c3.checkbox("IV rank filter (buy cheap IV)", False, key=k("ivr"))
        ivr_max = c3.slider("IV rank max", 0, 100, 30, key=k("ivrmax"))
        r_orb = c3.checkbox("30-min opening range (9:30–10:00)", False, key=k("orb"),
                            help="Calls: close above opening-range high. Puts: below the low. Uses Alpha Vantage 30-min bars (or Polygon with ORATS + FMP).")
        r_gap = c4.checkbox("Gap %", False, key=k("gap"))
        gap_min = c4.number_input("Min gap % (in trade direction)", 0.1, 20.0, 1.0, 0.1, key=k("gapmin"))
        r_rvol = c4.checkbox("Relative volume", False, key=k("rvol"))
        rvol_min = c4.number_input("Min RVOL (vs 20-day avg)", 0.5, 10.0, 1.5, 0.1, key=k("rvolmin"))
        EARN = ["Skip", "Hold through", "Exit before"]
        earn_mode = st.radio("Earnings", EARN, index=EARN.index(P["earn"]), horizontal=True, key=k("earn"),
                             help="Skip = no entry if earnings fall before the planned exit. "
                                  "Exit before = close the day before the report. LEAPS almost always span earnings, so Skip blocks them.")

        ivtag = st.checkbox("Tag each trade High / Low IV (results by IV regime)", True, key=k("ivtag"),
                            help="With Alpha Vantage this downloads one extra option chain per week of history "
                                 "(saved after the first run). Untick to make the first run faster.")

        st.subheader("Exit rules")
        c1, c2, c3, c4 = st.columns(4)
        pt = c1.number_input("Profit target % (0 = off)", 0, 1000, P["pt"], 5, key=k("pt"))
        sl = c2.number_input("Stop loss % (0 = off)", 0, 100, P["sl"], 5, key=k("sl"))
        ts = c3.number_input("Time stop: close at DTE (0 = off)", 0, 400, P["ts"], 1, key=k("ts"))
        hold = c4.number_input("Max days held (0 = off)", 0, 800, P["hold"], 1, key=k("hold"))
        c1, c2 = st.columns([1, 3])
        rev_on = c1.checkbox("Signal-reversal exit", True, key=k("rev"))
        rev_rule = c2.selectbox("Reversal rule", sig.REVERSAL_RULES,
                                index=sig.REVERSAL_RULES.index(P["rev"]), key=k("revrule"))

        st.subheader("Money & costs")
        c1, c2, c3, c4, c5 = st.columns(5)
        capital = c1.number_input("Starting capital $", 1_000, 50_000_000, 100_000, 5_000, key=k("cap"))
        alloc = c2.number_input("% of equity per trade", 0.5, 100.0, P["alloc"], 0.5, key=k("alloc"))
        maxpos = c3.number_input("Max open positions", 1, 50, 5, key=k("maxpos"))
        comm = c4.number_input("Commission $/contract/side", 0.0, 10.0, 0.65, 0.05, key=k("comm"))
        slip = c5.number_input("Extra slippage $/share", 0.0, 2.0, 0.0, 0.01, key=k("slip"),
                               help="Added on top of buying at the ask and selling at the bid.")
        go_btn = st.form_submit_button("▶ Run backtest", type="primary", use_container_width=True)

    st.caption("Timing: signals use the day's close → orders fill on the NEXT trading day at that day's end-of-day option prices "
               "(buy at ask, sell at bid, + commission). Stops/targets are checked once a day.")

    if go_btn:
        tickers = [t.strip().upper() for t in tickers_txt.replace(";", ",").split(",") if t.strip()]
        rules = sig.EntryRules(trend_pullback=r_pull, breakout=r_brk, breakout_len=int(brk_len),
                               ema_trend=r_ema, macd_cross=r_macd, macd_lookback=int(macd_lb),
                               ivr_filter=r_ivr, ivr_max=float(ivr_max), gap_filter=r_gap, gap_min=gap_min,
                               rvol_filter=r_rvol, rvol_min=rvol_min, orb_filter=r_orb)
        if not tickers:
            st.error("Add at least one ticker.")
        elif not rules.active():
            st.error("Tick at least one entry rule.")
        elif start >= end:
            st.error("Start date must be before end date.")
        else:
            params = engine.Params(
                tickers=tickers, start=pd.Timestamp(start), end=pd.Timestamp(end), strategy=preset,
                direction=direction, dte_min=dte[0], dte_max=dte[1], delta_min=delta[0], delta_max=delta[1],
                max_spread_pct=spread, min_oi=int(min_oi), profit_target=pt or None, stop_loss=sl or None,
                time_stop_dte=ts or None, max_hold_days=hold or None, reversal_exit=rev_on,
                reversal_rule=rev_rule, earnings_mode=earn_mode, commission=comm, slippage=slip,
                capital=float(capital), alloc_pct=alloc, max_positions=int(maxpos), rules=rules,
                iv_regime_tags=ivtag)
            bar = st.progress(0.0, "Starting…")
            try:
                res = engine.run(params, source, progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
                st.session_state["res"] = res
                st.session_state["synthetic"] = source.synthetic
                st.session_state.pop("ai_text", None)
                n = len(res.trades)
                st.success(f"Done: {n} trades from {F(res.equity.index[0])} to {F(res.equity.index[-1])}. "
                           "Open the 📊 Results tab.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
            finally:
                bar.empty()

res = st.session_state.get("res")

# --------------------------------------------------------------------------- results
with tab_res:
    if res is None:
        st.info("Run a backtest in ⚙️ Setup first.")
    else:
        p, tr, eq = res.params, res.trades, res.equity
        sm = mx.summary(tr, eq, p.capital)
        if st.session_state.get("synthetic"):
            st.markdown("<div class='warn'>⚠️ These results are from SYNTHETIC demo data.</div>", unsafe_allow_html=True)
        st.caption(f"{p.strategy} · {', '.join(p.tickers)} · {F(eq.index[0])} → {F(eq.index[-1])} · "
                   f"rules: {', '.join(p.rules.active())} · earnings: {p.earnings_mode}")

        money = lambda v: f"${v:,.0f}" if v >= 0 else f"−${-v:,.0f}"
        c = st.columns(4)
        c[0].metric("Trades", f"{sm.get('trades', 0):,}")
        c[1].metric("Win rate", f"{sm.get('win_rate', 0):.1f}%")
        c[2].metric("Expectancy / trade", money(sm.get("expectancy", 0)), f"{sm.get('expectancy_pct', 0):+.1f}% of premium")
        c[3].metric("Profit factor", "∞" if np.isinf(sm.get("profit_factor", 0)) else f"{sm.get('profit_factor', 0):.2f}")
        c = st.columns(4)
        c[0].metric("Total return", f"{sm.get('total_return_pct', 0):+.1f}%", f"{sm.get('cagr_pct', 0):+.1f}% CAGR")
        c[1].metric("Max drawdown", f"{sm.get('max_dd_pct', 0):.1f}%", help=f"Largest drop from a peak: {money(sm.get('max_dd_usd', 0))}")
        c[2].metric("Avg win", money(sm.get("avg_win", 0)), f"{sm.get('avg_win_pct', 0):+.1f}%")
        c[3].metric("Avg loss", money(sm.get("avg_loss", 0)), f"{sm.get('avg_loss_pct', 0):+.1f}%")

        st.plotly_chart(charts.equity(eq, p.capital), use_container_width=True, config={"displayModeBar": False})

        if tr.empty:
            st.warning("No trades. Loosen entry rules, widen delta/DTE, or raise the max spread.")
        else:
            views = {"Year": "year", "Ticker": "ticker", "Market regime (SPY vs 200-day)": "market",
                     "IV regime (IV rank ≥ 50 = high)": "iv_regime", "Calls vs puts": "type",
                     "Exit reason": "exit_reason"}
            pick = st.segmented_control("Break down by", list(views), default="Year") or "Year"
            bd = mx.breakdown(tr, views[pick])
            fig = charts.pnl_bars(bd, title=f"Total P&L by {pick.lower()}")
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(bd.style.format({"Win %": "{:.1f}", "Avg P&L $": "{:,.0f}", "Avg P&L %": "{:+.1f}",
                                          "Total P&L $": "{:,.0f}", "Avg days": "{:.0f}"}),
                         use_container_width=True)
            h = charts.pnl_hist(tr)
            if h:
                st.plotly_chart(h, use_container_width=True, config={"displayModeBar": False})
        with st.expander("Skipped signals & data notes"):
            st.write({k: v for k, v in res.skipped.items()})
            for n in res.notes[:40]:
                st.caption("• " + n)

# --------------------------------------------------------------------------- trades
with tab_trades:
    if res is None or res.trades.empty:
        st.info("No trades yet.")
    else:
        tr = res.trades
        show = pd.DataFrame({
            "Ticker": tr["ticker"], "Type": tr["type"].map({"C": "Call", "P": "Put"}),
            "Entry": tr["entry_date"].map(F), "Exit": tr["exit_date"].map(F), "Expiry": tr["expiry"].map(F),
            "Strike": tr["strike"], "DTE": tr["dte_entry"], "Delta": tr["delta_entry"].round(2),
            "Buy (ask)": tr["entry_px"].round(2), "Sell (bid)": tr["exit_px"].round(2),
            "Qty": tr["contracts"], "P&L $": tr["pnl"].round(0), "P&L %": tr["pnl_pct"].round(1),
            "Days": tr["days_held"], "Exit reason": tr["exit_reason"], "Market": tr["market"],
            "IV": tr["iv_regime"], "IV rank": tr["ivr_entry"].round(0)})
        st.dataframe(show, use_container_width=True, hide_index=True, height=420,
                     column_config={"P&L $": st.column_config.NumberColumn(format="$%d")})
        st.download_button("⬇ Download trades (CSV)", show.to_csv(index=False).encode(), "backtest_trades.csv",
                           "text/csv")
        st.subheader("Look at one trade")
        labels = [f"#{i + 1} {r.ticker} {('Call' if r.type == 'C' else 'Put')} {r.strike:g} · {F(r.entry_date)} → "
                  f"{F(r.exit_date)} · {r.pnl_pct:+.0f}%" for i, r in tr.iterrows()]
        ix = st.selectbox("Trade", range(len(labels)), format_func=lambda i: labels[i])
        t = tr.iloc[ix]
        try:
            st.plotly_chart(charts.trade_chart(source.prices(t["ticker"]), t), use_container_width=True,
                            config={"displayModeBar": False})
        except Exception as e:
            st.caption(f"Chart unavailable: {e}")

# --------------------------------------------------------------------------- AI
with tab_ai:
    if res is None:
        st.info("Run a backtest first.")
    elif not keys["ANTHROPIC_API_KEY"]:
        st.warning("Add ANTHROPIC_API_KEY to Streamlit Secrets to enable AI analysis.")
    else:
        sm = mx.summary(res.trades, res.equity, res.params.capital)
        q = st.text_input("Ask Claude (optional)", placeholder="e.g. Why did 2022 lose money? Should I only trade puts in bear markets?")
        if st.button("🤖 Explain results & suggest rule changes", type="primary", use_container_width=True):
            ctx = ai.build_context(res, sm)
            if st.session_state.get("synthetic"):
                ctx = "NOTE: THIS RUN USED SYNTHETIC DEMO DATA – say so.\n\n" + ctx
            st.session_state["ai_text"] = st.write_stream(ai.stream_analysis(ctx, q or None))
        elif st.session_state.get("ai_text"):
            st.markdown(st.session_state["ai_text"])
        st.caption("Claude sees your settings, the summary, every breakdown table and up to 150 trades. "
                   "Suggestions are ideas to re-test here — not trade advice.")
