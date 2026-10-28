"""Touch-friendly candlestick charts (TradingView Lightweight Charts, Apache-2.0, bundled in static/).

Every chart carries the same hard-set studies:
  * EMA 9 (blue) and EMA 200 (white) on price
  * Volume bars along the bottom of the price panel
  * MACD 12/26/9 in its own panel: MACD line red, signal line yellow, histogram

Pinch to zoom, drag to move, vertical swipes still scroll the page. Indicators are
computed on the full history and then trimmed, so EMA 200 is correct even on short views.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit.components.v1 as components

LIB = (Path(__file__).parent / "static" / "lightweight-charts.standalone.production.js").read_text()

UP, DOWN = "#0ca30c", "#d03b3b"
EMA9_COLOR, EMA200_COLOR = "#3b82f6", "#ffffff"
MACD_COLOR, SIGNAL_COLOR = "#ef4444", "#facc15"
LINE_STYLE = {"solid": 0, "dot": 1, "dash": 2, "dashdot": 3}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA 9, EMA 200 and MACD (12, 26, 9) columns."""
    df = df.copy()
    c = df["Close"].astype(float)
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()
    df.loc[df.index[:199], "EMA200"] = np.nan          # not meaningful until 200 bars exist
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD"] = macd
    df["Signal"] = macd.ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    df.loc[df.index[:25], ["MACD", "Signal", "Hist"]] = np.nan
    return df


def _times(index: pd.Index) -> list[int]:
    """Chart times are New York wall-clock written as UTC seconds, so the axis reads in ET."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    # seconds since 1970 – computed this way it doesn't depend on pandas' internal time unit
    return ((idx - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64").tolist()


def _series(times, values, color=None):
    out = []
    for t, v in zip(times, values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append({"time": t})                      # whitespace keeps both panels aligned
        else:
            item = {"time": t, "value": round(float(v), 6)}
            if color:
                item["color"] = color(v)
            out.append(item)
    return out


def _segments(lines) -> list[dict]:
    """Split each extra line at missing values into separate pieces: the chart engine would
    otherwise join a line straight across a gap (e.g. VWAP across the overnight session)."""
    out = []
    for ln in lines:
        pieces, cur = [], []
        for t, v in ln["points"]:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                if cur:
                    pieces.append(cur)
                cur = []
            else:
                cur.append({"time": to_time(t), "value": float(v)})
        if cur:
            pieces.append(cur)
        for n, pts in enumerate(pieces):
            out.append({"color": ln.get("color", "#94a3b8"), "style": LINE_STYLE.get(ln.get("style", "solid"), 0),
                        "title": ln.get("title", "") if n == len(pieces) - 1 else "", "points": pts})
    return out


def to_time(ts) -> int:
    return _times(pd.DatetimeIndex([ts]))[0]


def render(bars: pd.DataFrame, *, key: str, title: str = "", height: int = 460,
           intraday: bool = True, style: str = "Filled", levels=(), markers=(), lines=(),
           note: str = "") -> None:
    """Draw candles + EMA9/EMA200 + volume + MACD.

    bars    : Open/High/Low/Close/Volume (+ EMA9/EMA200/MACD/Signal/Hist from add_indicators)
    levels  : [{"price", "title", "color", "style"}]  horizontal lines with axis labels
    markers : [{"time": Timestamp, "up": bool, "text"}] breakout arrows
    lines   : [{"points": [(Timestamp, price), ...], "color", "style", "title"}] extra lines
    """
    if bars.empty:
        components.html(f"<div style='color:#fbbf24;font:14px system-ui;padding:8px'>"
                        f"⚠️ {title}: no data – check the symbol.</div>", height=40)
        return
    if "EMA9" not in bars:
        bars = add_indicators(bars)
    times = _times(bars.index)
    mins = (pd.DatetimeIndex(bars.index).hour * 60 + pd.DatetimeIndex(bars.index).minute).tolist()
    hollow = style == "Hollow"
    candles = []
    for t, m, o, h, l, c in zip(times, mins, bars["Open"], bars["High"], bars["Low"], bars["Close"]):
        item = {"time": t, "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
        if intraday and not (570 <= m < 960):            # dim pre-market / after-hours candles
            col = "rgba(12,163,12,0.45)" if c >= o else "rgba(208,59,59,0.45)"
            item.update(color="rgba(0,0,0,0)" if hollow and c >= o else col, borderColor=col, wickColor=col)
        candles.append(item)
    vol = [{"time": t, "value": float(v or 0),
            "color": "rgba(12,163,12,0.35)" if c >= o else "rgba(208,59,59,0.35)"}
           for t, v, o, c in zip(times, bars["Volume"].fillna(0), bars["Open"], bars["Close"])]
    payload = {
        "title": title, "note": note, "intraday": intraday, "hollow": hollow, "key": key,
        "candles": candles, "volume": vol,
        "ema9": _series(times, bars["EMA9"]), "ema200": _series(times, bars["EMA200"]),
        "macd": _series(times, bars["MACD"]), "signal": _series(times, bars["Signal"]),
        "hist": _series(times, bars["Hist"],
                        lambda v: "rgba(12,163,12,0.55)" if v >= 0 else "rgba(208,59,59,0.55)"),
        "levels": [{"price": float(lv["price"]), "title": lv.get("title", ""),
                    "color": lv.get("color", "#94a3b8"), "style": LINE_STYLE.get(lv.get("style", "dash"), 2),
                    "label": lv.get("label", True)} for lv in levels],
        "markers": sorted([{"time": to_time(mk["time"]), "position": "belowBar" if mk["up"] else "aboveBar",
                            "shape": "arrowUp" if mk["up"] else "arrowDown",
                            "color": UP if mk["up"] else DOWN, "text": mk.get("text", "")}
                           for mk in markers], key=lambda m: m["time"]),
        "lines": _segments(lines),
        "colors": {"up": UP, "down": DOWN, "ema9": EMA9_COLOR, "ema200": EMA200_COLOR,
                   "macd": MACD_COLOR, "signal": SIGNAL_COLOR},
    }
    price_h, macd_h = height, max(120, int(height * 0.3))
    components.html(_HTML.replace("__LIB__", LIB)
                    .replace("__DATA__", json.dumps(payload))
                    .replace("__PH__", str(price_h)).replace("__MH__", str(macd_h)),
                    height=price_h + macd_h + 74)


_HTML = r"""
<style>
  html,body{margin:0;background:#0b0f19;color:#cbd5e1;font:12px Inter,system-ui,sans-serif;overflow:hidden}
  #top{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:2px 2px 4px;flex-wrap:wrap}
  #ttl{font-weight:700;font-size:14px;color:#f3f4f6}
  #legend{display:flex;gap:10px;flex-wrap:wrap;font:11px ui-monospace,monospace;color:#94a3b8}
  .sw{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:4px}
  button{background:#111827;color:#cbd5e1;border:1px solid #1f2937;border-radius:6px;padding:3px 8px;font:11px system-ui;cursor:pointer}
  #price,#macd{width:100%;position:relative}
  #mlabel{position:absolute;left:6px;top:2px;z-index:3;font:10px ui-monospace,monospace;color:#94a3b8;pointer-events:none}
  #note{font:11px system-ui;color:#94a3b8;padding-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
</style>
<div id="top">
  <span id="ttl"></span>
  <span id="legend">
    <span><i class="sw" style="background:#3b82f6"></i>EMA 9 <b id="v9"></b></span>
    <span><i class="sw" style="background:#ffffff"></i>EMA 200 <b id="v200"></b></span>
    <span>Vol <b id="vv"></b></span>
  </span>
  <button id="reset" title="Show the most recent candles again">Reset view</button>
</div>
<div id="price" style="height:__PH__px"></div>
<div id="macd" style="height:__MH__px"><div id="mlabel">MACD 12·26·9 <i class="sw" style="background:#ef4444"></i>MACD <b id="vm"></b> <i class="sw" style="background:#facc15"></i>Signal <b id="vs"></b></div></div>
<div id="note"></div>
<script>__LIB__</script>
<script>
const D = __DATA__;
document.getElementById('ttl').textContent = D.title;
document.getElementById('note').textContent = D.note || '';
const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const pad = n => String(n).padStart(2,'0');
const fmtDate = d => pad(d.getUTCDate()) + '/' + MON[d.getUTCMonth()] + '/' + d.getUTCFullYear();
const fmtTime = d => pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes());
const fmtFull = t => { const d = new Date(t*1000); return D.intraday ? fmtDate(d) + ' ' + fmtTime(d) : fmtDate(d); };
const tick = (t, type) => { const d = new Date(t*1000);
  if (type === 0) return String(d.getUTCFullYear());                       // year
  if (type === 1) return MON[d.getUTCMonth()] + ' ' + d.getUTCFullYear();    // month
  if (type === 2) return pad(d.getUTCDate()) + '/' + MON[d.getUTCMonth()];  // day
  return fmtTime(d); };
const fmtVol = v => v >= 1e9 ? (v/1e9).toFixed(2)+'B' : v >= 1e6 ? (v/1e6).toFixed(2)+'M' : v >= 1e3 ? (v/1e3).toFixed(1)+'K' : String(Math.round(v));

const base = {
  autoSize: true,
  layout: { background: { color: '#0b0f19' }, textColor: '#cbd5e1', fontFamily: 'Inter, system-ui, sans-serif' },
  grid: { vertLines: { color: '#1a2230' }, horzLines: { color: '#1a2230' } },
  rightPriceScale: { borderColor: '#1f2937' },
  timeScale: { borderColor: '#1f2937', timeVisible: D.intraday, secondsVisible: false, rightOffset: 4,
               tickMarkFormatter: tick },
  localization: { timeFormatter: fmtFull },
  crosshair: { mode: 0 },
  // touch: drag sideways to move, pinch to zoom; vertical swipes keep scrolling the page
  handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
  handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: { time: true, price: true },
                 axisDoubleClickReset: true },
  kineticScroll: { touch: true, mouse: false },
};
const pc = LightweightCharts.createChart(document.getElementById('price'), base);
const mc = LightweightCharts.createChart(document.getElementById('macd'),
  Object.assign({}, base, { timeScale: Object.assign({}, base.timeScale, { visible: true }) }));

const C = D.colors;
const cs = pc.addCandlestickSeries({
  upColor: D.hollow ? 'rgba(0,0,0,0)' : C.up, downColor: C.down,
  borderUpColor: C.up, borderDownColor: C.down, wickUpColor: C.up, wickDownColor: C.down,
  priceLineVisible: true, priceLineStyle: 2 });
cs.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.24 } });
cs.setData(D.candles);
const vs = pc.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol',
  lastValueVisible: false, priceLineVisible: false });
vs.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
vs.setData(D.volume);
const line = (chart, color, width, data, title) => { const s = chart.addLineSeries({ color, lineWidth: width,
  lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, title: title || '' });
  s.setData(data); return s; };
const e9 = line(pc, C.ema9, 2, D.ema9);
const e200 = line(pc, C.ema200, 2, D.ema200);
D.levels.forEach(l => cs.createPriceLine({ price: l.price, color: l.color, lineWidth: 1, lineStyle: l.style,
  axisLabelVisible: l.label, title: l.title }));
D.lines.forEach(l => { const s = pc.addLineSeries({ color: l.color, lineWidth: 2, lineStyle: l.style,
  lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, title: l.title });
  s.setData(l.points); });
if (D.markers.length) cs.setMarkers(D.markers);

const hs = mc.addHistogramSeries({ lastValueVisible: false, priceLineVisible: false });
hs.setData(D.hist);
const ms = line(mc, C.macd, 2, D.macd);
const ss = line(mc, C.signal, 2, D.signal);

// keep the two panels scrolled/zoomed together
let syncing = false;
const sync = (from, to) => from.timeScale().subscribeVisibleLogicalRangeChange(r => {
  if (syncing || !r) return; syncing = true; to.timeScale().setVisibleLogicalRange(r); syncing = false; save(); });
sync(pc, mc); sync(mc, pc);

// live legend values under the crosshair (latest bar when not hovering)
const last = arr => { for (let i = arr.length - 1; i >= 0; i--) if (arr[i].value !== undefined) return arr[i].value; };
const setLegend = (v9, v200, vv, vm, vsg) => {
  document.getElementById('v9').textContent = v9 !== undefined ? v9.toFixed(2) : '–';
  document.getElementById('v200').textContent = v200 !== undefined ? v200.toFixed(2) : '–';
  document.getElementById('vv').textContent = vv !== undefined ? fmtVol(vv) : '–';
  document.getElementById('vm').textContent = vm !== undefined ? vm.toFixed(3) : '–';
  document.getElementById('vs').textContent = vsg !== undefined ? vsg.toFixed(3) : '–'; };
const latest = () => setLegend(last(D.ema9), last(D.ema200), last(D.volume), last(D.macd), last(D.signal));
latest();
pc.subscribeCrosshairMove(p => {
  if (!p.time || !p.seriesData) { latest(); return; }
  const g = s => { const d = p.seriesData.get(s); return d && d.value !== undefined ? d.value : undefined; };
  const i = D.candles.findIndex(c => c.time === p.time);
  setLegend(g(e9), g(e200), g(vs), i >= 0 ? D.macd[i].value : undefined, i >= 0 ? D.signal[i].value : undefined);
  try { mc.setCrosshairPosition(i >= 0 && D.macd[i].value !== undefined ? D.macd[i].value : 0, p.time, ms); } catch (e) {}
});
pc.subscribeCrosshairMove(p => { if (!p.time) { try { mc.clearCrosshairPosition(); } catch (e) {} } });

// start on the most recent candles; remember the user's zoom across auto-refreshes
const N = D.candles.length, KEY = 'tv_view_' + D.key;
function defaultView() {
  const show = Math.min(N, D.intraday ? 160 : 180);
  pc.timeScale().setVisibleLogicalRange({ from: N - show, to: N + 3 });
}
function save() {
  if (typeof touched !== 'undefined' && !touched) return;
  try { const r = pc.timeScale().getVisibleLogicalRange(); if (!r) return;
        localStorage.setItem(KEY, JSON.stringify({ right: N - 1 - r.to, width: r.to - r.from, n: N })); } catch (e) {}
}
function restore() {
  try { const v = JSON.parse(localStorage.getItem(KEY) || 'null');
        if (v && v.width > 5) { const to = N - 1 - v.right; pc.timeScale().setVisibleLogicalRange({ from: to - v.width, to }); return true; }
  } catch (e) {}
  return false;
}
// The page lays the chart out at a small width first and widens it a moment later. Until the
// user touches the chart, re-fit the view whenever its size changes, so candles fill the width.
let touched = false;
['pointerdown', 'touchstart', 'wheel'].forEach(ev =>
  ['price', 'macd'].forEach(id => document.getElementById(id).addEventListener(ev, () => { touched = true; }, { passive: true })));
const fit = () => { if (!restore()) defaultView(); };
fit();
let lastW = 0;
new ResizeObserver(() => {
  const w = document.getElementById('price').clientWidth;
  if (w > 0 && w !== lastW) { lastW = w; if (!touched) requestAnimationFrame(fit); }
}).observe(document.getElementById('price'));
document.getElementById('reset').onclick = () => { try { localStorage.removeItem(KEY); } catch (e) {} touched = false; defaultView(); };
</script>
"""
