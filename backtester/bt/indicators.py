"""Daily indicators. Every value on row t uses only data up to and including day t's close."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig


def atr(df: pd.DataFrame, n=14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def features(px: pd.DataFrame, ivr: pd.Series | None = None, orb: pd.DataFrame | None = None,
             breakout_len: int = 20) -> pd.DataFrame:
    f = px.copy()
    c = f["close"]
    f["ema9"], f["ema21"], f["ema50"], f["ema200"] = ema(c, 9), ema(c, 21), ema(c, 50), ema(c, 200)
    f["macd"], f["macd_sig"] = macd(c)
    f["atr"] = atr(f)
    f["gap_pct"] = (f["open"] / c.shift() - 1) * 100
    f["rvol"] = f["volume"] / f["volume"].shift().rolling(20, min_periods=20).mean()
    f["hh"] = f["high"].shift().rolling(breakout_len, min_periods=breakout_len).max()
    f["ll"] = f["low"].shift().rolling(breakout_len, min_periods=breakout_len).min()
    f["ivr"] = ivr.reindex(f.index) if ivr is not None and len(ivr) else np.nan
    if orb is not None and len(orb):
        o = orb.reindex(f.index)
        f["orh"], f["orl"] = o["orh"], o["orl"]
    else:
        f["orh"] = f["orl"] = np.nan
    return f
