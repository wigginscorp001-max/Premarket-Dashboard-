"""Entry / exit rules. All rules are evaluated on day t's CLOSE; trades happen on day t+1."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EntryRules:
    trend_pullback: bool = True      # uptrend + dip to EMA 9 then reclaim
    breakout: bool = False           # close beyond N-day high/low
    breakout_len: int = 20
    ema_trend: bool = False          # EMA 9 vs EMA 200 alignment
    macd_cross: bool = False         # MACD 12/26/9 crossed signal within N bars
    macd_lookback: int = 3
    ivr_filter: bool = False         # only buy when IV rank is cheap
    ivr_max: float = 30.0
    gap_filter: bool = False         # gap % in trade direction
    gap_min: float = 1.0
    rvol_filter: bool = False        # relative volume
    rvol_min: float = 1.5
    orb_filter: bool = False         # close beyond 9:30–10:00 opening range (needs Polygon)

    def active(self) -> list[str]:
        names = {"trend_pullback": "Trend pullback", "breakout": "Breakout", "ema_trend": "EMA 9 / 200",
                 "macd_cross": "MACD cross", "ivr_filter": "IV rank filter", "gap_filter": "Gap %",
                 "rvol_filter": "Relative volume", "orb_filter": "Opening range"}
        return [v for k, v in names.items() if getattr(self, k)]


def _crossed(a: pd.Series, b: pd.Series, up: bool, lookback: int) -> pd.Series:
    x = (a > b) & (a.shift() <= b.shift()) if up else (a < b) & (a.shift() >= b.shift())
    return x.astype(int).rolling(max(lookback, 1), min_periods=1).max().astype(bool)


def entry_signals(f: pd.DataFrame, r: EntryRules) -> tuple[pd.Series, pd.Series]:
    """Returns (call_signal, put_signal) boolean series on the signal day."""
    c = f["close"]
    long_parts, short_parts = [], []

    if r.trend_pullback:
        up = (c > f["ema200"]) & (f["ema50"] > f["ema200"])
        dn = (c < f["ema200"]) & (f["ema50"] < f["ema200"])
        long_parts.append(up & (c.shift() < f["ema9"].shift()) & (c > f["ema9"]))
        short_parts.append(dn & (c.shift() > f["ema9"].shift()) & (c < f["ema9"]))
    if r.breakout:
        long_parts.append(c > f["hh"])
        short_parts.append(c < f["ll"])
    if r.ema_trend:
        long_parts.append((f["ema9"] > f["ema200"]) & (c > f["ema9"]))
        short_parts.append((f["ema9"] < f["ema200"]) & (c < f["ema9"]))
    if r.macd_cross:
        long_parts.append(_crossed(f["macd"], f["macd_sig"], True, r.macd_lookback))
        short_parts.append(_crossed(f["macd"], f["macd_sig"], False, r.macd_lookback))
    if r.ivr_filter:
        cheap = f["ivr"] <= r.ivr_max
        long_parts.append(cheap)
        short_parts.append(cheap)
    if r.gap_filter:
        long_parts.append(f["gap_pct"] >= r.gap_min)
        short_parts.append(f["gap_pct"] <= -r.gap_min)
    if r.rvol_filter:
        long_parts.append(f["rvol"] >= r.rvol_min)
        short_parts.append(f["rvol"] >= r.rvol_min)
    if r.orb_filter:
        long_parts.append(c > f["orh"])
        short_parts.append(c < f["orl"])

    if not long_parts:
        false = pd.Series(False, index=f.index)
        return false, false
    longs = np.logical_and.reduce([p.fillna(False).astype(bool) for p in long_parts])
    shorts = np.logical_and.reduce([p.fillna(False).astype(bool) for p in short_parts])
    warm = f["ema200"].notna()
    return pd.Series(longs, f.index) & warm, pd.Series(shorts, f.index) & warm


REVERSAL_RULES = ["Close crosses EMA 9", "MACD crosses signal", "Close crosses EMA 200"]


def reversal_signals(f: pd.DataFrame, rule: str) -> tuple[pd.Series, pd.Series]:
    """(exit_calls, exit_puts): the trend has turned against the position."""
    c = f["close"]
    if rule == "MACD crosses signal":
        a, b = f["macd"], f["macd_sig"]
        return (a < b) & (a.shift() >= b.shift()), (a > b) & (a.shift() <= b.shift())
    line = f["ema200"] if rule == "Close crosses EMA 200" else f["ema9"]
    return (c < line).fillna(False), (c > line).fillna(False)
