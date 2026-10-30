"""Market sentiment gauge (fear ↔ greed, 0–100) and 52-week high/low stats.

The gauge averages whichever of these readings are available; each is scored 0–100
(0 = extreme fear, 50 = neutral, 100 = extreme greed):

  * Options money  – market tide: net call premium vs net put premium today
  * Put/call       – SPY put volume ÷ call volume (high = fear)
  * Momentum       – SPY vs its 50-day average
  * 52-week range  – where SPY sits between its 52-week low and high
  * Sector breadth – share of the 11 sector ETFs up today
  * Volatility     – SPY IV rank (high implied volatility = fear)
  * Dealer gamma   – SPY gamma regime (negative = bigger, fear-driven moves)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = [(25, "Extreme Fear"), (45, "Fear"), (55, "Neutral"), (75, "Greed"), (101, "Extreme Greed")]


def label(score: float) -> str:
    return next(name for top, name in LABELS if score < top)


def _clip(v: float) -> float:
    return float(np.clip(v, 0, 100))


def _col(df: pd.DataFrame, *names):
    return next((n for n in names if n in df.columns), None)


def tide_score(tide: pd.DataFrame | None) -> tuple[float, str] | None:
    if tide is None or tide.empty:
        return None
    c, p = _col(tide, "net_call_premium"), _col(tide, "net_put_premium")
    if not (c and p):
        return None
    nc, npp = float(pd.to_numeric(tide[c]).iloc[-1]), float(pd.to_numeric(tide[p]).iloc[-1])
    tot = abs(nc) + abs(npp)
    if not tot:
        return None
    return _clip(50 + 50 * (nc - npp) / tot), f"net calls ${nc / 1e6:,.0f}M vs net puts ${npp / 1e6:,.0f}M"


def put_call_score(vol: pd.DataFrame | None) -> tuple[float, str] | None:
    if vol is None or vol.empty:
        return None
    c, p = _col(vol, "call_volume"), _col(vol, "put_volume")
    if not (c and p):
        return None
    cv, pv = float(pd.to_numeric(vol[c]).iloc[0]), float(pd.to_numeric(vol[p]).iloc[0])
    if not cv:
        return None
    pc = pv / cv
    return _clip(50 - (pc - 0.9) / 0.5 * 50), f"SPY put/call {pc:.2f}"


def momentum_score(daily: pd.DataFrame) -> tuple[float, str] | None:
    c = daily["Close"].dropna() if daily is not None and len(daily) else pd.Series(dtype=float)
    if len(c) < 50:
        return None
    pct = (c.iloc[-1] / c.tail(50).mean() - 1) * 100
    return _clip(50 + pct / 5 * 50), f"SPY {pct:+.1f}% vs 50-day average"


def range_score(daily: pd.DataFrame) -> tuple[float, str] | None:
    s = week52(daily)
    if not s:
        return None
    return _clip(s["Range %"]), f"SPY at {s['Range %']:.0f}% of its 52-week range"


def breadth_score(changes: list[float]) -> tuple[float, str] | None:
    ch = [c for c in changes if c == c]
    if not ch:
        return None
    up = sum(c > 0 for c in ch)
    return 100 * up / len(ch), f"{up} of {len(ch)} sectors up"


def iv_score(ivr: pd.DataFrame | None) -> tuple[float, str] | None:
    if ivr is None or ivr.empty:
        return None
    col = _col(ivr, "iv_rank_1y", "iv_rank", "rank")
    if not col:
        return None
    v = float(pd.to_numeric(ivr[col]).iloc[-1])
    v = v * 100 if v <= 1 else v
    return _clip(100 - v), f"SPY IV rank {v:.0f}"


def gamma_score(regime: str | None) -> tuple[float, str] | None:
    if regime not in ("Positive", "Negative"):
        return None
    return (65.0, "SPY gamma positive (calmer)") if regime == "Positive" else (35.0, "SPY gamma negative (bigger moves)")


def gauge(parts: dict[str, tuple[float, str] | None]) -> tuple[float, pd.DataFrame]:
    """(overall score, table of the readings used)."""
    rows = [{"Reading": k, "Score": round(v[0]), "Mood": label(v[0]), "Detail": v[1]}
            for k, v in parts.items() if v is not None]
    df = pd.DataFrame(rows, columns=["Reading", "Score", "Mood", "Detail"])
    return (float(df["Score"].mean()) if len(df) else float("nan")), df


def week52(daily: pd.DataFrame, price: float = float("nan")) -> dict:
    """52-week high/low (with dates), distance from each and position in the range."""
    if daily is None or daily.empty:
        return {}
    d = daily[daily.index >= daily.index[-1] - pd.Timedelta(days=365)]
    hi_i, lo_i = d["High"].idxmax(), d["Low"].idxmin()
    hi, lo = float(d["High"].max()), float(d["Low"].min())
    px = price if price == price else float(d["Close"].iloc[-1])
    if px > hi:                                 # today's live price can set a new extreme
        hi, hi_i = px, pd.Timestamp.now().normalize()
    if px < lo:
        lo, lo_i = px, pd.Timestamp.now().normalize()
    return {"Price": px, "52w high": hi, "High date": hi_i, "52w low": lo, "Low date": lo_i,
            "From high %": (px / hi - 1) * 100, "From low %": (px / lo - 1) * 100,
            "Range %": (px - lo) / (hi - lo) * 100 if hi > lo else 50.0}
