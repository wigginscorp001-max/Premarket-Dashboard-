"""Scanner: ranks tickers by gap, relative volume, pre-market / opening-range breaks,
options flow and gamma levels.

Five signals are checked per ticker; each one points up (bullish) or down (bearish):
  1. Gap        – |gap %| at or above the minimum
  2. Volume     – relative volume at or above the minimum (counts for either direction)
  3. PM range   – above the pre-market high / below the pre-market low
  4. ORB        – above the 9:30–10:00 high / below its low (after 10:00)
  5. Flow       – unusual options premium leaning to calls / puts
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import session as ses

NAN = float("nan")
PM_START, OPEN, ORB_END, CLOSE = 240, 570, 600, 960   # minutes after midnight, ET


def _minutes(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour * 60 + idx.minute


# --------------------------------------------------------------------------- #
# Relative volume
# --------------------------------------------------------------------------- #
def relative_volume(bars: pd.DataFrame, max_days: int = 10) -> float:
    """Volume since 4:00 today vs. the average volume by the same time on prior days.

    2.0 means twice as much trading as usual by this time of day.
    """
    if bars is None or bars.empty:
        return NAN
    m = _minutes(bars.index)
    bars = bars[m >= PM_START]
    if bars.empty:
        return NAN
    dates = bars.index.date
    days = sorted(set(dates))
    today = days[-1]
    cutoff = _minutes(bars.index[dates == today])[-1]
    upto = _minutes(bars.index) <= cutoff
    vol_today = bars["Volume"][(dates == today) & upto].sum()
    prior = [bars["Volume"][(dates == d) & upto].sum() for d in days[-1 - max_days:-1]]
    prior = [v for v in prior if v > 0]
    if len(prior) < 2:
        return NAN
    return float(vol_today / np.mean(prior))


# --------------------------------------------------------------------------- #
# Options flow (Unusual Whales flow alerts)
# --------------------------------------------------------------------------- #
def flow_summary(alerts: pd.DataFrame | None, today=None) -> pd.DataFrame:
    """Per ticker: call premium, put premium, number of alerts, % of premium in calls.

    Uses one day of alerts: `today`, or the latest day in the data when not given.
    """
    cols = ["Calls $", "Puts $", "Alerts", "Calls %"]
    if alerts is None or alerts.empty:
        return pd.DataFrame(columns=cols)
    df = alerts
    tcol = next((c for c in ("ticker", "ticker_symbol", "underlying_symbol") if c in df.columns), None)
    pcol = next((c for c in ("total_premium", "premium") if c in df.columns), None)
    kcol = next((c for c in ("type", "option_type") if c in df.columns), None)
    if not (tcol and pcol and kcol):
        return pd.DataFrame(columns=cols)
    if "created_at" in df.columns:
        # options only trade 9:30-4:00, so before the open use the latest session's alerts
        ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
        day = today if today is not None else ts.dt.date.max()
        df = df[ts.dt.date == day]
    if df.empty:
        return pd.DataFrame(columns=cols)
    side = df[kcol].astype(str).str.lower().str[0]          # "c" / "p"
    prem = pd.to_numeric(df[pcol], errors="coerce").fillna(0)
    out = pd.DataFrame({
        "Calls $": prem.where(side == "c", 0).groupby(df[tcol]).sum(),
        "Puts $": prem.where(side == "p", 0).groupby(df[tcol]).sum(),
        "Alerts": df.groupby(tcol).size(),
    })
    total = out["Calls $"] + out["Puts $"]
    out["Calls %"] = (out["Calls $"] / total.replace(0, np.nan) * 100)
    return out


# --------------------------------------------------------------------------- #
# Gamma levels (Unusual Whales spot exposures by strike)
# --------------------------------------------------------------------------- #
def gamma_levels(df: pd.DataFrame | None, spot: float = NAN, band: float = 0.10) -> dict:
    """Call wall, put wall, gamma flip and regime from gamma-by-strike data.

    * Call wall – strike at or above the price with the most call gamma (often resistance)
    * Put wall  – strike at or below the price with the most put gamma (often support)
    * Regime    – sign of net gamma around the current price: Positive = dealers damp moves
                  (choppy, pinned), Negative = dealers add to moves (trending, bigger ranges)
    * Flip      – nearest price where that net gamma changes sign; an estimate
    Walls are looked for within ±band of the price (2x that for tickers with wide strike gaps).
    """
    out = {"Call wall": NAN, "Put wall": NAN, "Gamma flip": NAN, "Gamma": "—"}
    if df is None or df.empty or "strike" not in df.columns:
        return out
    cg = next((c for c in ("call_gamma_oi", "call_gex", "call_gamma") if c in df.columns), None)
    pg = next((c for c in ("put_gamma_oi", "put_gex", "put_gamma") if c in df.columns), None)
    if not (cg and pg):
        return out
    d = df.copy()
    if "time" in d.columns:      # each strike carries its own update time: keep each strike's latest row
        d = d.sort_values("time").drop_duplicates("strike", keep="last")
    if not np.isfinite(spot) and "price" in d.columns:
        px = pd.to_numeric(d["price"], errors="coerce").dropna()
        spot = float(px.iloc[-1]) if len(px) else NAN
    for c in ("strike", cg, pg):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    call = d[cg].fillna(0).abs()
    put = d[pg].fillna(0).abs()                               # size of put gamma
    net = (call - put).to_numpy()                             # puts count as negative gamma
    strikes = d["strike"].to_numpy()
    if not np.isfinite(spot) or spot <= 0 or (call.sum() == 0 and put.sum() == 0):
        return out

    for name, size, side in (("Call wall", call, strikes >= spot), ("Put wall", put, strikes <= spot)):
        # look on the wall's own side of the price; widen the search for tickers with wide strike gaps
        for reach in (band, 2 * band, np.inf):
            pick = side & (np.abs(strikes / spot - 1) <= reach)
            if size[pick].sum() > 0:
                out[name] = float(strikes[size[pick].idxmax()])
                break

    # Net gamma profile across nearby prices: each strike's net gamma spread as a bell curve
    # (~3% of the price wide), summed. Its sign at the price is the regime; where it crosses
    # zero is the flip.
    w = 0.03 * spot
    grid = np.linspace(spot * (1 - 2 * band), spot * (1 + 2 * band), 401)
    use = np.abs(strikes / spot - 1) <= 0.3
    prof = (net[use][None, :] * np.exp(-((grid[:, None] - strikes[use][None, :]) / w) ** 2)).sum(axis=1)
    at_spot = float(np.interp(spot, grid, prof))
    out["Gamma"] = "Positive" if at_spot >= 0 else "Negative"
    cross = np.where(np.sign(prof[:-1]) * np.sign(prof[1:]) < 0)[0]
    if len(cross):
        g0, g1, p0, p1 = grid[cross], grid[cross + 1], prof[cross], prof[cross + 1]
        flips = g0 + (g1 - g0) * (-p0 / (p1 - p0))
        out["Gamma flip"] = round(float(flips[np.argmin(np.abs(flips - spot))]), 2)
    return out


# --------------------------------------------------------------------------- #
# One row per ticker
# --------------------------------------------------------------------------- #
def scan_row(ticker: str, bars: pd.DataFrame, flow: pd.Series | None = None, gamma: dict | None = None,
             *, min_gap: float = 2.0, min_rvol: float = 2.0, min_flow: float = 100_000) -> dict:
    base = ses.today(ticker, bars)
    row = {"Ticker": ticker}
    if bars is None or bars.empty or "PM high" not in base:
        row.update({"Score": 0, "Direction": "—", "Signals": base.get("Behavior", "No data")})
        row.update(gamma or {})
        return row
    prev = base.get("Prev close", NAN)
    price = float(bars["Close"].iloc[-1])
    last_min = int(_minutes(bars.index[-1:])[0])
    is_today_rth = last_min >= OPEN and "Open" in base
    gap = (base["Open"] / prev - 1) * 100 if is_today_rth and prev == prev else base.get("Gap %", NAN)
    gap_usd = base["Open"] - prev if is_today_rth and prev == prev else base.get("Gap $", NAN)
    rv = relative_volume(bars)
    pmh, pml = base["PM high"], base["PM low"]

    # VWAP: regular-session VWAP after the open, pre-market VWAP before it
    if is_today_rth:
        day = bars[bars.index.date == bars.index[-1].date()]
        rth = day[(_minutes(day.index) >= OPEN) & (_minutes(day.index) < CLOSE)]
        v = rth["Volume"].sum()
        vwap = float(((rth["High"] + rth["Low"] + rth["Close"]) / 3 * rth["Volume"]).sum() / v) if v else NAN
    else:
        vwap = base.get("PM VWAP", NAN)

    bull, bear, hits = 0, 0, []
    if gap == gap and abs(gap) >= min_gap:
        hits.append(f"Gap {gap:+.1f}%")
        bull, bear = bull + (gap > 0), bear + (gap < 0)
    if rv == rv and rv >= min_rvol:
        hits.append(f"RVOL {rv:.1f}x")
        bull, bear = bull + 1, bear + 1
    if is_today_rth:
        pm_status = "Above PM high" if price > pmh else "Below PM low" if price < pml else "Inside PM range"
    else:
        rng = pmh - pml
        pm_status = ("At PM high" if rng > 0 and price >= pmh - 0.1 * rng else
                     "At PM low" if rng > 0 and price <= pml + 0.1 * rng else "Mid PM range")
    if pm_status in ("Above PM high", "At PM high"):
        hits.append(pm_status)
        bull += 1
    elif pm_status in ("Below PM low", "At PM low"):
        hits.append(pm_status)
        bear += 1
    orh, orl = base.get("ORB high", NAN), base.get("ORB low", NAN)
    if "ORB break" in base:      # after 10:00
        orb_status = "Above ORB high" if price > orh else "Below ORB low" if price < orl else "Inside ORB"
        if orb_status == "Above ORB high":
            hits.append("Above ORB high")
            bull += 1
        elif orb_status == "Below ORB low":
            hits.append("Below ORB low")
            bear += 1
    else:
        orb_status = "Forming" if "ORB high" in base else "Before 9:30"

    calls = puts = calls_pct = NAN
    n_alerts = 0
    if flow is not None:
        calls, puts, n_alerts, calls_pct = (flow.get("Calls $", NAN), flow.get("Puts $", NAN),
                                            int(flow.get("Alerts", 0)), flow.get("Calls %", NAN))
        if (calls + puts) >= min_flow and calls_pct == calls_pct:
            if calls_pct >= 60:
                hits.append(f"Flow {calls_pct:.0f}% calls")
                bull += 1
            elif calls_pct <= 40:
                hits.append(f"Flow {100 - calls_pct:.0f}% puts")
                bear += 1

    if bull > bear:
        direction, score = "▲ Bullish", bull
    elif bear > bull:
        direction, score = "▼ Bearish", bear
    else:   # equal pull both ways (e.g. gap down, but trading above the PM and opening-range highs)
        direction, score = "◆ Mixed", bull
    row.update({
        "Score": int(score), "Direction": direction if score else "—",
        "Price": price, "Change %": (price / prev - 1) * 100 if prev == prev else NAN,
        "Gap %": gap, "Gap $": gap_usd, "RVOL": rv,
        "PM high": pmh, "PM low": pml, "PM status": pm_status,
        "ORB high": orh, "ORB low": orl, "ORB status": orb_status,
        "VWAP": vwap, "vs VWAP": ("Above" if price >= vwap else "Below") if vwap == vwap else "—",
        "Calls $": calls, "Puts $": puts, "Calls %": calls_pct, "Flow alerts": n_alerts,
        "Signals": " · ".join(hits) if hits else "—",
    })
    row.update(gamma or {})
    return row


# --------------------------------------------------------------------------- #
# Confluence card: 8 checks that either agree or disagree
# --------------------------------------------------------------------------- #
def confluence(row: dict, bars: pd.DataFrame, *, min_gap: float = 2.0, min_rvol: float = 2.0,
               min_flow: float = 100_000) -> dict:
    """Eight quick checks for one ticker, each bullish (+1), bearish (-1) or neutral (0).

    `row` is a scan_row() result; `bars` are the same 5-minute bars with EMA 9 / EMA 200 /
    MACD columns (tvchart.add_indicators). Relative volume doesn't pick a side: when it is
    high it backs whichever side is ahead.
    """
    checks: list[tuple[int, str]] = []
    price = row.get("Price", NAN)
    last = bars.iloc[-1] if bars is not None and len(bars) else None

    def vs_line(name: str, col: str):
        v = last.get(col, NAN) if last is not None else NAN
        if not (v == v and price == price):
            checks.append((0, f"{name} not ready"))
        else:
            checks.append((1, f"Above {name} ({v:,.2f})") if price >= v else (-1, f"Below {name} ({v:,.2f})"))

    vs_line("EMA 9", "EMA9")
    vs_line("EMA 200", "EMA200")
    vwap = row.get("VWAP", NAN)
    if vwap == vwap and price == price:
        checks.append((1, f"Above VWAP ({vwap:,.2f})") if price >= vwap else (-1, f"Below VWAP ({vwap:,.2f})"))
    else:
        checks.append((0, "VWAP not ready"))
    m, s = (last.get("MACD", NAN), last.get("Signal", NAN)) if last is not None else (NAN, NAN)
    if m == m and s == s:
        checks.append((1, "MACD above signal") if m >= s else (-1, "MACD below signal"))
    else:
        checks.append((0, "MACD not ready"))
    gap = row.get("Gap %", NAN)
    if gap == gap and abs(gap) >= min_gap:
        checks.append((1 if gap > 0 else -1, f"Gap {'up' if gap > 0 else 'down'} {gap:+.1f}%"))
    else:
        checks.append((0, f"Small gap {gap:+.1f}%" if gap == gap else "No gap data"))
    rv = row.get("RVOL", NAN)
    vol_strong = rv == rv and rv >= min_rvol
    vol_text = (f"Rel. volume {rv:.1f}x" if vol_strong else
                f"Light volume {rv:.1f}x" if rv == rv else "Volume not ready")
    orb, pm = row.get("ORB status", ""), row.get("PM status", "")
    if orb in ("Above ORB high", "Below ORB low"):
        checks.append((1 if orb.startswith("Above") else -1, orb))
    elif pm in ("Above PM high", "Below PM low", "At PM high", "At PM low"):
        checks.append((1 if "high" in pm else -1, pm))
    else:
        checks.append((0, orb if orb == "Inside ORB" else pm or "No breakout yet"))
    calls_pct = row.get("Calls %", NAN)
    total = (row.get("Calls $", 0) or 0) + (row.get("Puts $", 0) or 0)
    if calls_pct == calls_pct and total >= min_flow and calls_pct >= 60:
        checks.append((1, f"Flow {calls_pct:.0f}% calls"))
    elif calls_pct == calls_pct and total >= min_flow and calls_pct <= 40:
        checks.append((-1, f"Flow {100 - calls_pct:.0f}% puts"))
    elif calls_pct == calls_pct and total >= min_flow:
        checks.append((0, f"Flow mixed ({calls_pct:.0f}% calls)"))
    else:
        checks.append((0, "No unusual flow"))

    bull = sum(1 for v, _ in checks if v > 0)
    bear = sum(1 for v, _ in checks if v < 0)
    lead = 1 if bull > bear else -1 if bear > bull else 0
    checks.insert(5, ((lead if vol_strong else 0), vol_text))     # volume backs the leader
    bull, bear = bull + (vol_strong and lead > 0), bear + (vol_strong and lead < 0)
    direction = "▲ Bullish" if lead > 0 else "▼ Bearish" if lead < 0 else "◆ Mixed"

    # Gamma context (not scored): regime and how close the walls are
    notes = []
    if row.get("Gamma") in ("Positive", "Negative"):
        notes.append(f"{row['Gamma']} gamma – " + ("choppy / pinned" if row["Gamma"] == "Positive"
                                                   else "moves can run"))
    for name in ("Call wall", "Put wall"):
        w = row.get(name, NAN)
        if w == w and price == price and price:
            d = (w / price - 1) * 100
            notes.append(f"{name} {w:,.2f} ({d:+.1f}%)" + (" ⚠️ close" if abs(d) <= 1 else ""))
    return {"Ticker": row["Ticker"], "Price": price, "Change %": row.get("Change %", NAN),
            "Direction": direction, "Bull": int(bull), "Bear": int(bear), "Total": len(checks),
            "Checks": checks, "Notes": notes}
