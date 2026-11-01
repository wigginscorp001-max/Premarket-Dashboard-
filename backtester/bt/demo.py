"""SYNTHETIC test data – only for checking the app works before API keys are added.

Prices are random walks; option prices are Black-Scholes with a bid/ask spread.
Results from demo mode mean nothing for real trading.
"""
from __future__ import annotations

import zlib
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import ndtr

R = 0.04
COLS = ["expiry", "dte", "strike", "spot", "type", "bid", "ask", "delta", "oi", "iv"]


def _rng(ticker: str, salt: str = "") -> np.random.Generator:
    return np.random.default_rng(zlib.crc32((ticker.upper() + salt).encode()))


def _business_days(start="2008-01-01", end=None) -> pd.DatetimeIndex:
    end = end or pd.Timestamp.today().normalize()
    return pd.bdate_range(start, end)


@lru_cache(maxsize=64)
def daily_prices(ticker: str) -> pd.DataFrame:
    rng = _rng(ticker)
    idx = _business_days()
    n = len(idx)
    # regime-switching drift / vol
    regime = np.sign(np.sin(np.arange(n) / rng.uniform(180, 320) + rng.uniform(0, 6)))
    mu = np.where(regime > 0, 0.0007, -0.0004)
    vol = _iv(ticker.upper()).values / np.sqrt(252) * 0.9
    ret = mu + vol * rng.standard_normal(n)
    close = 50 * np.exp(np.cumsum(ret)) * rng.uniform(0.6, 4)
    gap = 0.3 * vol * rng.standard_normal(n)
    open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
    hi = np.maximum(open_, close) * (1 + np.abs(vol * 0.6 * rng.standard_normal(n)))
    lo = np.minimum(open_, close) * (1 - np.abs(vol * 0.6 * rng.standard_normal(n)))
    volu = rng.lognormal(15, 0.4, n) * (1 + 3 * np.abs(ret) / vol.mean())
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": volu},
                        index=idx)


def iv_path(ticker: str, idx=None) -> pd.Series:
    return _iv(ticker.upper()).reindex(idx) if idx is not None else _iv(ticker.upper())


@lru_cache(maxsize=64)
def _iv(ticker: str) -> pd.Series:
    idx = _business_days()
    rng = _rng(ticker, "iv")
    x = np.zeros(len(idx))
    for i in range(1, len(idx)):
        x[i] = 0.97 * x[i - 1] + 0.05 * rng.standard_normal()
    return pd.Series(0.28 * np.exp(x * 0.9), index=idx).clip(0.1, 1.2)


def iv_rank(ticker: str) -> pd.Series:
    iv = iv_path(ticker)
    lo, hi = iv.rolling(252, min_periods=60).min(), iv.rolling(252, min_periods=60).max()
    return 100 * (iv - lo) / (hi - lo)


def earnings_dates(ticker: str) -> list[pd.Timestamp]:
    idx = _business_days()
    off = int(_rng(ticker, "e").integers(0, 63))
    return list(idx[off::63])


def _expiries(date: pd.Timestamp, max_days: int) -> list[pd.Timestamp]:
    out = []
    m = pd.Timestamp(date.year, date.month, 1)
    for _ in range(int(max_days / 28) + 3):
        third_fri = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-FRI")[2]
        if third_fri > date:
            out.append(third_fri)
        m = m + pd.offsets.MonthBegin(1)
    return out


def chain(ticker: str, date: pd.Timestamp, dte_min: int, dte_max: int) -> pd.DataFrame:
    px = daily_prices(ticker)
    date = pd.Timestamp(date)
    if date not in px.index:
        return pd.DataFrame(columns=COLS)
    s = float(px.at[date, "close"])
    sig = float(iv_path(ticker).get(date, 0.3))
    step = 1.0 if s < 60 else 2.5 if s < 200 else 5.0
    strikes = np.arange(np.floor(s * 0.6 / step) * step, s * 1.4, step)
    exps = [ex for ex in _expiries(date, dte_max) if dte_min <= (ex - date).days <= dte_max]
    if not exps:
        return pd.DataFrame(columns=COLS)
    ex_arr = np.repeat(np.array(exps, dtype="datetime64[ns]"), len(strikes))
    k = np.tile(strikes, len(exps))
    dte = (ex_arr - np.datetime64(date, "ns")).astype("timedelta64[D]").astype(int)
    t = dte / 365
    d1 = (np.log(s / k) + (R + sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    c = s * ndtr(d1) - k * np.exp(-R * t) * ndtr(d2)
    p = k * np.exp(-R * t) * ndtr(-d2) - s * ndtr(-d1)
    mid = np.r_[c, p]
    half = np.maximum(0.025, mid * 0.015)
    n = len(k)
    return pd.DataFrame({"expiry": np.r_[ex_arr, ex_arr], "dte": np.r_[dte, dte], "strike": np.r_[k, k],
                         "spot": s, "type": np.array(["C"] * n + ["P"] * n, dtype=object),
                         "bid": np.maximum(mid - half, 0), "ask": mid + half,
                         "delta": np.r_[ndtr(d1), ndtr(d1) - 1], "oi": 500.0, "iv": sig})


def opening_range(ticker, start, end) -> pd.DataFrame:
    px = daily_prices(ticker)
    rng = _rng(ticker, "orb")
    w = rng.uniform(0.2, 0.6, len(px))
    orh = px["open"] + w * (px["high"] - px["open"])
    orl = px["open"] - w * (px["open"] - px["low"])
    return pd.DataFrame({"orh": orh, "orl": orl}, index=px.index)
