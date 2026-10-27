"""Chart-pattern detection on daily (or intraday) candles, plus how each pattern
has played out historically on the same ticker.

Every detector is vectorized, so it marks every bar in the history where the
pattern was present. That gives two things at once: whether the pattern is
there *today*, and how price moved N bars after it appeared in the past.

Patterns are geometric approximations (trend lines fitted by least squares),
not hand-drawn chart reads. Treat them as a shortlist to look at, not a signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import atr, rsi

BIAS = {
    "Bull flag": "Bullish", "Bear flag": "Bearish",
    "Falling wedge": "Bullish", "Rising wedge": "Bearish",
    "Ascending triangle": "Bullish", "Descending triangle": "Bearish",
    "Symmetrical triangle": "Neutral",
    "Rising channel": "Bullish", "Falling channel": "Bearish",
    "Breakout (20-day high)": "Bullish", "Breakdown (20-day low)": "Bearish",
    "Double bottom": "Bullish", "Double top": "Bearish",
    "Oversold (RSI<30)": "Bullish", "Overbought (RSI>70)": "Bearish",
    "Inside bar": "Neutral", "NR7 (tightest range in 7)": "Neutral",
}


# --------------------------------------------------------------------------- #
# Rolling least-squares lines
# --------------------------------------------------------------------------- #
def _rolling_line(y: pd.Series, w: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Slope, fitted start value and fitted end value of a line fit over the last w bars."""
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    sx, sy = x.rolling(w).sum(), y.rolling(w).sum()
    sxy, sxx = (x * y).rolling(w).sum(), (x * x).rolling(w).sum()
    slope = (w * sxy - sx * sy) / (w * sxx - sx ** 2)
    mid = sy / w
    half = (w - 1) / 2
    return slope, mid - slope * half, mid + slope * half


def _envelope(h: pd.Series, l: pd.Series, w: int) -> dict[str, pd.Series]:
    """Upper/lower trend lines through the extremes of three equal segments of the window.

    Using segment highs/lows (instead of fitting every bar) tracks the touch points
    a trader would draw a line through, so zig-zagging price doesn't flatten the lines.
    """
    s = w // 3
    hi = [h.rolling(s).max().shift(k * s) for k in (2, 1, 0)]   # oldest -> newest segment
    lo = [l.rolling(s).min().shift(k * s) for k in (2, 1, 0)]
    up_s, lo_s = (hi[2] - hi[0]) / (2 * s), (lo[2] - lo[0]) / (2 * s)
    return {
        "up_s": up_s, "lo_s": lo_s,
        "up_start": hi[0], "lo_start": lo[0],
        "up_end": hi[2] + up_s * s / 2, "lo_end": lo[2] + lo_s * s / 2,
        # how far the middle segment sits from a straight line (0 = perfectly straight)
        "bend": ((hi[1] - (hi[0] + hi[2]) / 2).abs() + (lo[1] - (lo[0] + lo[2]) / 2).abs()) / 2,
    }


def _first_of_run(sig: pd.Series) -> pd.Series:
    """Only the first bar of each consecutive run (so one pattern = one event)."""
    sig = sig.fillna(False).astype(bool)
    return sig & ~sig.shift(1, fill_value=False)


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def detect(df: pd.DataFrame, w: int = 30) -> dict[str, pd.Series]:
    """Boolean series per pattern, True on bars where it is present."""
    h, l, c = df["High"], df["Low"], df["Close"]
    a = atr(df)
    flat = 0.04 * a          # slope per bar that counts as "flat"
    e = _envelope(h, l, w)
    up_s, lo_s, up_end, lo_end = e["up_s"], e["lo_s"], e["up_end"], e["lo_end"]
    width_start, width_end = e["up_start"] - e["lo_start"], up_end - lo_end
    straight = e["bend"] < 0.75 * a
    converging = straight & (width_end < 0.65 * width_start) & (width_end > 0)
    parallel = straight & (width_end / width_start).between(0.7, 1.4)
    # price still inside the pattern (not already broken out)
    inside = (c <= up_end + 0.25 * a) & (c >= lo_end - 0.25 * a)

    out: dict[str, pd.Series] = {
        "Rising wedge": converging & inside & (up_s > flat) & (lo_s > up_s),
        "Falling wedge": converging & inside & (lo_s < -flat) & (up_s < lo_s),
        "Ascending triangle": converging & inside & (up_s.abs() <= flat) & (lo_s > flat),
        "Descending triangle": converging & inside & (lo_s.abs() <= flat) & (up_s < -flat),
        "Symmetrical triangle": converging & inside & (up_s < -flat) & (lo_s > flat),
        "Rising channel": parallel & inside & (up_s > 2 * flat) & (lo_s > 2 * flat),
        "Falling channel": parallel & inside & (up_s < -2 * flat) & (lo_s < -2 * flat),
    }

    # Flags: a sharp pole, then a tight, slightly counter-trend pause.
    pole_n, flag_n = 8, 7
    pole = c.shift(flag_n) - c.shift(flag_n + pole_n)
    pole_atr = pole / a.shift(flag_n)
    f_hi, f_lo = h.rolling(flag_n).max(), l.rolling(flag_n).min()
    f_slope, _, _ = _rolling_line(c, flag_n)
    tight = (f_hi - f_lo) < 0.5 * pole.abs()
    out["Bull flag"] = (pole_atr > 3.5) & tight & (f_slope <= 0.02 * a) & (c > c.shift(flag_n) - 0.5 * pole)
    out["Bear flag"] = (pole_atr < -3.5) & tight & (f_slope >= -0.02 * a) & (c < c.shift(flag_n) - 0.5 * pole)

    hi20, lo20 = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    out["Breakout (20-day high)"] = c > hi20
    out["Breakdown (20-day low)"] = c < lo20

    # Double top / bottom: two similar extremes 10-40 bars apart with a real dip/rally between.
    look = 40
    peak = h.rolling(look).max()
    peak_age = h.rolling(look).apply(lambda s: len(s) - 1 - np.argmax(s), raw=True)
    trough = l.rolling(look).min()
    trough_age = l.rolling(look).apply(lambda s: len(s) - 1 - np.argmin(s), raw=True)
    recent_hi, recent_lo = h.rolling(5).max(), l.rolling(5).min()
    out["Double top"] = ((peak_age >= 10) & ((recent_hi - peak).abs() <= 0.5 * a)
                         & (l.rolling(look).min() < peak - 2.5 * a) & (c < recent_hi - 0.3 * a))
    out["Double bottom"] = ((trough_age >= 10) & ((recent_lo - trough).abs() <= 0.5 * a)
                            & (h.rolling(look).max() > trough + 2.5 * a) & (c > recent_lo + 0.3 * a))

    r = rsi(c)
    out["Oversold (RSI<30)"] = r < 30
    out["Overbought (RSI>70)"] = r > 70
    rng = h - l
    out["Inside bar"] = (h < h.shift()) & (l > l.shift())
    out["NR7 (tightest range in 7)"] = rng <= rng.rolling(7).min()
    return {k: v.fillna(False).astype(bool) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def scan(ticker: str, df: pd.DataFrame, horizon: int = 10) -> tuple[list[dict], pd.DataFrame]:
    """(patterns present on the latest bar, historical stats for every pattern)."""
    sigs = detect(df)
    fwd = (df["Close"].shift(-horizon) / df["Close"] - 1) * 100
    stats, today = [], []
    for name, sig in sigs.items():
        events = _first_of_run(sig)
        moves = fwd[events].dropna()
        bias = BIAS[name]
        if bias == "Bullish":
            hit = (moves > 0).mean() * 100 if len(moves) else np.nan
        elif bias == "Bearish":
            hit = (moves < 0).mean() * 100 if len(moves) else np.nan
        else:
            hit = np.nan
        row = {"Pattern": name, "Bias": bias, "Times seen": int(len(moves)),
               f"Worked % ({horizon}d)": hit, f"Avg move % ({horizon}d)": moves.mean() if len(moves) else np.nan,
               "Avg abs move %": moves.abs().mean() if len(moves) else np.nan}
        stats.append(row)
        if bool(sig.iloc[-1]):
            run = int(sig[::-1].cumprod().sum())  # how many bars it has been present
            today.append({"Ticker": ticker, **row, "Bars in pattern": run})
    return today, pd.DataFrame(stats)


def overall_bias(found: list[dict]) -> str:
    score = sum({"Bullish": 1, "Bearish": -1}.get(f["Bias"], 0) for f in found)
    return "Bullish" if score > 0 else "Bearish" if score < 0 else ("Neutral" if found else "—")


def pattern_lines(df: pd.DataFrame, name: str, w: int = 30) -> list[tuple]:
    """Trend lines to draw for the latest bar's pattern: [(x0, y0, x1, y1, label), ...]."""
    tail = df.tail(w)
    if name in ("Bull flag", "Bear flag"):
        tail = df.tail(7)
    elif name.startswith(("Breakout", "Breakdown")):
        lvl = df["High"].iloc[-21:-1].max() if name.startswith("Breakout") else df["Low"].iloc[-21:-1].min()
        return [(df.index[-21], lvl, df.index[-1], lvl, "20-day level")]
    elif name not in ("Rising wedge", "Falling wedge", "Ascending triangle", "Descending triangle",
                      "Symmetrical triangle", "Rising channel", "Falling channel"):
        return []
    s = w // 3
    seg = [tail.iloc[k * s:(k + 1) * s] for k in range(3)]
    lines = []
    for col, fn, lbl in (("High", "idxmax", "upper"), ("Low", "idxmin", "lower")):
        p0, p2 = getattr(seg[0][col], fn)(), getattr(seg[2][col], fn)()
        y0, y2 = seg[0][col][p0], seg[2][col][p2]
        i0, i2 = df.index.get_loc(p0), df.index.get_loc(p2)
        slope = (y2 - y0) / max(i2 - i0, 1)
        lines.append((p0, y0, df.index[-1], y0 + slope * (len(df) - 1 - i0), lbl))
    return lines
