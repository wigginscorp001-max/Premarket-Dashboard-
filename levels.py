"""Key price levels and multi-timeframe bars for the chart grid.

Levels that "lock" do so because of the bars they are built from:
  * pre-market high/low use only 4:00-9:29 bars, so they stop changing at 9:30;
  * the opening range uses only 9:30-9:59 bars, so it is final at 10:00.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from analytics import atr, day_levels

DATE_FMT = "%d/%b/%Y"          # 28/Oct/2026
DATETIME_FMT = "%d/%b/%Y %H:%M"

# label -> (bar source, resample rule or None, how much history to show)
TIMEFRAMES = {
    "1m": ("1m", None, "session"),
    "2m": ("1m", "2min", "session"),
    "3m": ("1m", "3min", "2 sessions"),
    "5m": ("5m", None, "2 sessions"),
    "10m": ("5m60", "10min", "7 days"),
    "15m": ("5m60", "15min", "14 days"),
    "30m": ("5m60", "30min", "30 days"),
    "1h": ("5m60", "60min", "60 days"),
    "Daily": ("daily", None, "6 months"),
    "Weekly": ("daily", "W-FRI", "3 years"),
    "YTD": ("daily", None, "ytd"),
    "1Y": ("daily", None, "12 months"),
    "Max": ("dailymax", "W-FRI", "all"),
}
INTRADAY = {"1m", "2m", "3m", "5m", "10m", "15m", "30m", "1h"}

# Custom range: candle size -> (bar source, resample rule, how far back that source reaches in days)
CUSTOM_CANDLES = {
    "1m": ("1m7", None, 7), "2m": ("1m7", "2min", 7), "3m": ("1m7", "3min", 7),
    "5m": ("5m60", None, 60), "10m": ("5m60", "10min", 60), "15m": ("5m60", "15min", 60),
    "30m": ("5m60", "30min", 60), "1h": ("5m60", "60min", 60),
    "Daily": ("dailymax", None, None), "Weekly": ("dailymax", "W-FRI", None),
}


def fmt_date(d) -> str:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "—"
    return pd.Timestamp(d).strftime(DATE_FMT)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Hourly candles start at :30 so the first regular-session bar is 9:30-10:30.
    offset = "30min" if rule == "60min" else None
    out = df.resample(rule, label="left", closed="left", offset=offset).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return out.dropna(subset=["Open"])


def window(df: pd.DataFrame, span: str) -> pd.DataFrame:
    """Trim bars to the look-back a timeframe shows."""
    if df.empty:
        return df
    if span.endswith("session") or span.endswith("sessions"):
        n = 1 if span == "session" else int(span.split()[0])
        days = sorted(set(df.index.date))[-n:]
        out = df[np.isin(df.index.date, days)]
        # Before the 4:00 pre-market starts there is nothing for "today" yet: show yesterday too.
        return out if len(out) >= 10 or n > 1 else df[np.isin(df.index.date, sorted(set(df.index.date))[-2:])]
    if span == "all":
        return df
    end = df.index[-1]
    if span == "ytd":
        return df[df.index >= pd.Timestamp(end.year, 1, 1, tz=end.tz)]
    qty, unit = span.split()
    delta = {"days": pd.Timedelta(days=int(qty)), "months": pd.DateOffset(months=int(qty)),
             "years": pd.DateOffset(years=int(qty))}[unit]
    return df[df.index >= end - delta]


# --------------------------------------------------------------------------- #
# Support / resistance from swing points
# --------------------------------------------------------------------------- #
def swing_levels(bars: pd.DataFrame, k: int = 3, lookback: int = 120,
                 merge_atr: float = 0.5) -> list[dict]:
    """Prices where the market turned at least once, merged when close together.

    A swing high is a bar whose high is the highest of the k bars on each side
    (a swing low likewise). Swings within `merge_atr` x ATR of each other count
    as one level; `touches` is how many swings formed it.
    """
    d = bars.tail(lookback)
    if len(d) < 2 * k + 5:
        return []
    a = float(atr(d).iloc[-1])
    win = 2 * k + 1
    hi = d["High"][d["High"] == d["High"].rolling(win, center=True).max()]
    lo = d["Low"][d["Low"] == d["Low"].rolling(win, center=True).min()]
    pts = sorted(list(hi.values) + list(lo.values))
    levels: list[dict] = []
    for p in pts:
        if levels and p - levels[-1]["last"] <= merge_atr * a:
            lv = levels[-1]
            lv["sum"] += p
            lv["touches"] += 1
            lv["last"] = p
        else:
            levels.append({"sum": p, "touches": 1, "last": p})
    return [{"price": lv["sum"] / lv["touches"], "touches": lv["touches"]} for lv in levels]


def nearest_sr(levels: list[dict], price: float, n: int = 2) -> tuple[list[dict], list[dict]]:
    """(n nearest resistances above price, n nearest supports below), closest first."""
    above = sorted([lv for lv in levels if lv["price"] > price], key=lambda lv: lv["price"])[:n]
    below = sorted([lv for lv in levels if lv["price"] <= price], key=lambda lv: -lv["price"])[:n]
    return above, below


# --------------------------------------------------------------------------- #
# Everything for one ticker
# --------------------------------------------------------------------------- #
def ticker_levels(intra5: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Intraday levels (PM, opening range, prior day, pivots) + daily support/resistance."""
    if intra5 is None or intra5.empty or daily is None or len(daily) < 2:
        raise ValueError("no data")
    base = day_levels(intra5, daily)
    lv, today, rth = base["levels"], base["today"], base["rth"]
    price = float(today["Close"].iloc[-1]) if len(today) else float(daily["Close"].iloc[-1])
    res, sup = nearest_sr(swing_levels(daily), price)
    now_min = today.index[-1].hour * 60 + today.index[-1].minute if len(today) else 0
    out = {"price": price, "levels": lv, "resistance": res, "support": sup,
           "pm_locked": now_min >= 570, "orb_locked": now_min >= 600, "breaks": []}
    if "ORH" in lv and len(rth):
        after = rth[rth.index.time >= dt.time(10, 0)]
        for side, col, test in (("up", "High", lambda s: s > lv["ORH"]),
                                ("down", "Low", lambda s: s < lv["ORL"])):
            hits = after[test(after[col])]
            if len(hits):
                t = hits.index[0]
                out["breaks"].append({"side": side, "time": t,
                                      "price": lv["ORH"] if side == "up" else lv["ORL"]})
    return out


def long_term_sr(daily: pd.DataFrame, weekly: bool) -> list[dict]:
    """Support/resistance for daily and weekly charts."""
    if weekly:
        return swing_levels(resample(daily, "W-FRI"), k=2, lookback=156, merge_atr=0.6)
    return swing_levels(daily, k=3, lookback=250, merge_atr=0.5)
