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

    * Call wall – strike with the most call gamma (often acts as resistance)
    * Put wall  – strike with the most put gamma (often acts as support)
    * Flip      – where cumulative net gamma (low strike → high) crosses zero; an estimate
    * Regime    – Positive above the flip (dealers damp moves: choppy, pinned),
                  Negative below it (dealers add to moves: trending, bigger ranges)
    Only strikes within ±band of the price are used.
    """
    out = {"Call wall": NAN, "Put wall": NAN, "Gamma flip": NAN, "Gamma": "—"}
    if df is None or df.empty or "strike" not in df.columns:
        return out
    cg = next((c for c in ("call_gamma_oi", "call_gex", "call_gamma") if c in df.columns), None)
    pg = next((c for c in ("put_gamma_oi", "put_gex", "put_gamma") if c in df.columns), None)
    if not (cg and pg):
        return out
    d = df.copy()
    if "time" in d.columns and d["time"].nunique() > 1:      # several snapshots: keep the latest
        d = d[d["time"] == d["time"].max()]
    if "price" in d.columns and not np.isfinite(spot):
        spot = float(pd.to_numeric(d["price"], errors="coerce").dropna().iloc[-1]) if d["price"].notna().any() else NAN
    for c in ("strike", cg, pg):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["strike"]).sort_values("strike")
    call = d[cg].fillna(0).abs()
    put = -d[pg].fillna(0).abs()                              # puts count as negative gamma
    if np.isfinite(spot) and spot > 0:
        keep = d["strike"].between(spot * (1 - band), spot * (1 + band))
        d, call, put = d[keep], call[keep], put[keep]
    if d.empty or (call.sum() == 0 and put.sum() == 0):
        return out
    out["Call wall"] = float(d["strike"][call.idxmax()]) if call.max() > 0 else NAN
    out["Put wall"] = float(d["strike"][put.idxmin()]) if put.min() < 0 else NAN
    net = (call + put).to_numpy()
    cum = np.cumsum(net)
    strikes = d["strike"].to_numpy()
    cross = np.where(np.sign(cum[:-1]) != np.sign(cum[1:]))[0]
    if len(cross):
        flips = strikes[cross + 1]
        flip = float(flips[np.argmin(np.abs(flips - spot))]) if np.isfinite(spot) else float(flips[0])
        out["Gamma flip"] = flip
        if np.isfinite(spot):
            out["Gamma"] = "Positive" if spot >= flip else "Negative"
    if out["Gamma"] == "—":
        out["Gamma"] = "Positive" if net.sum() >= 0 else "Negative"
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

    if bull > bear or (bull == bear and gap == gap and gap > 0):
        direction, score = "▲ Bullish", bull
    elif bear > bull or (bull == bear and gap == gap and gap < 0):
        direction, score = "▼ Bearish", bear
    else:
        direction, score = "—", bull
    row.update({
        "Score": int(score), "Direction": direction if score else "—",
        "Price": price, "Change %": (price / prev - 1) * 100 if prev == prev else NAN,
        "Gap %": gap, "RVOL": rv,
        "PM high": pmh, "PM low": pml, "PM status": pm_status,
        "ORB high": orh, "ORB low": orl, "ORB status": orb_status,
        "VWAP": vwap, "vs VWAP": ("Above" if price >= vwap else "Below") if vwap == vwap else "—",
        "Calls $": calls, "Puts $": puts, "Calls %": calls_pct, "Flow alerts": n_alerts,
        "Signals": " · ".join(hits) if hits else "—",
    })
    row.update(gamma or {})
    return row
