"""Alpha Vantage (premium): one feed for everything the backtester needs.

  HISTORICAL_OPTIONS          full chain for one symbol on one date (2008+): bid/ask, IV, greeks, OI
  TIME_SERIES_DAILY_ADJUSTED  25+ yrs daily OHLCV + split coefficients
  EARNINGS                    past report dates
  TIME_SERIES_INTRADAY        30-min bars by month (first regular-hours bar = 9:30–10:00 opening range)

IV rank is not offered, so it is built from weekly ATM implied volatility (1 extra download per week
of history, cached). Requests are throttled to ALPHAVANTAGE_RPM (default 75 = the $49.99 plan).
"""
from __future__ import annotations

import datetime as dt
import threading
import time

import numpy as np
import pandas as pd
import requests

from . import store
from .config import secret

BASE = "https://www.alphavantage.co/query"
CHAIN_COLS = ["expiry", "dte", "strike", "spot", "type", "bid", "ask", "delta", "oi", "iv"]
_session = requests.Session()
_lock = threading.Lock()
_last = [0.0]


class AVError(RuntimeError):
    pass


def _rpm() -> int:
    try:
        return max(5, int(secret("ALPHAVANTAGE_RPM") or 75))
    except ValueError:
        return 75


def _throttle():
    gap = 60.0 / _rpm() * 1.05
    with _lock:
        wait = _last[0] + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()


def _get(function: str, **params):
    key = secret("ALPHAVANTAGE_API_KEY")
    if not key:
        raise AVError("ALPHAVANTAGE_API_KEY is missing from Streamlit Secrets.")
    params.update(function=function, apikey=key)
    for attempt in range(6):
        _throttle()
        r = _session.get(BASE, params=params, timeout=90)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict):
            if "Error Message" in js:
                raise AVError(f"Alpha Vantage: {js['Error Message']}")
            msg = js.get("Information") or js.get("Note")
            if msg and len(js) <= 2:
                if "premium" in msg.lower() and "rate" not in msg.lower():
                    raise AVError(f"Alpha Vantage: {msg}")
                time.sleep(15 * (attempt + 1))        # per-minute limit hit – wait and retry
                continue
        return js
    raise AVError(f"Alpha Vantage kept refusing {function} {params.get('symbol')} {params.get('date', '')}")


def _today() -> dt.date:
    return dt.datetime.utcnow().date()


# ----------------------------------------------------------------------------- prices
def _daily_raw(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()
    key = f"av:daily:{ticker}"
    hit = store.get(key)
    if hit is not None and hit[1].date() >= _today():
        return hit[0]
    js = _get("TIME_SERIES_DAILY_ADJUSTED", symbol=ticker, outputsize="full")
    ts = js.get("Time Series (Daily)") or {}
    if not ts:
        if hit is not None:
            return hit[0]
        raise AVError(f"Alpha Vantage returned no daily prices for {ticker}.")
    df = pd.DataFrame.from_dict(ts, orient="index").rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close",
        "6. volume": "volume", "8. split coefficient": "split"})
    df = df[["open", "high", "low", "close", "volume", "split"]].astype(float)
    df.index.name = "date"
    df = df.reset_index()
    store.put(key, df)
    return df


def daily_prices(ticker: str) -> pd.DataFrame:
    """Split-adjusted OHLCV (like FMP) plus close_raw (as traded, for matching option strikes)."""
    df = _daily_raw(ticker).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    split = df["split"].replace(0, 1.0).fillna(1.0)
    # price on day t is divided by the product of all split coefficients AFTER t
    later = split[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out = pd.DataFrame(index=df.index)
    for c in ("open", "high", "low", "close"):
        out[c] = df[c] / later
    out["volume"] = df["volume"] * later
    out["close_raw"] = df["close"]
    return out


def earnings_dates(ticker: str) -> list[pd.Timestamp]:
    ticker = ticker.upper()
    key = f"av:earn:{ticker}"
    hit = store.get(key)
    if hit is not None and hit[1].date() >= _today():
        df = hit[0]
    else:
        js = _get("EARNINGS", symbol=ticker)
        q = js.get("quarterlyEarnings") or []
        df = pd.DataFrame({"date": [x.get("reportedDate") for x in q if x.get("reportedDate")]})
        store.put(key, df)
    if df.empty:
        return []
    return sorted(set(pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize()))


# ----------------------------------------------------------------------------- options
def normalise(rows: list[dict], date: pd.Timestamp, dte_min: int = 0, dte_max: int = 5000) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    if raw.empty or "strike" not in raw.columns:
        return pd.DataFrame(columns=CHAIN_COLS)
    expiry = pd.to_datetime(raw["expiration"]).dt.normalize()
    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else np.nan
    out = pd.DataFrame({
        "expiry": expiry,
        "dte": (expiry - pd.Timestamp(date).normalize()).dt.days,
        "strike": num("strike"),
        "spot": np.nan,                                 # not in the chain; engine uses daily prices
        "type": raw["type"].str.lower().map({"call": "C", "put": "P"}),
        "bid": num("bid"), "ask": num("ask"), "delta": num("delta"),
        "oi": num("open_interest"), "iv": num("implied_volatility"),
    })
    out = out[out["type"].notna() & (out["dte"] >= dte_min) & (out["dte"] <= dte_max)]
    return out[CHAIN_COLS].reset_index(drop=True)


def chain(ticker: str, date: pd.Timestamp, dte_min: int, dte_max: int) -> pd.DataFrame:
    ticker = ticker.upper()
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    key = f"av:chain:{ticker}:{d}:{dte_min}-{dte_max}"
    hit = store.get(key)
    if hit is not None:                      # history never changes – cache forever
        return hit[0]
    js = _get("HISTORICAL_OPTIONS", symbol=ticker, date=d)
    df = normalise(js.get("data") or [], pd.Timestamp(date), dte_min, dte_max)
    store.put(key, df)
    return df


def _atm_iv(ch: pd.DataFrame) -> float:
    """Average IV of near-the-money (|delta| 0.35–0.65) contracts in the expiry closest to 30 DTE."""
    if ch is None or ch.empty:
        return np.nan
    exp = ch.loc[(ch["dte"] - 30).abs().idxmin(), "expiry"]
    c = ch[(ch["expiry"] == exp) & ch["delta"].abs().between(0.35, 0.65) & (ch["iv"] > 0)]
    return float(c["iv"].median()) if len(c) else np.nan


def iv_rank(ticker: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
            progress=None) -> pd.Series:
    """1-year IV rank (0–100) from weekly ATM IV, forward-filled to daily. Only uses data known by each day."""
    px = daily_prices(ticker)
    end = pd.Timestamp(end or px.index[-1])
    start = pd.Timestamp(start or px.index[0]) - pd.Timedelta(days=380)   # 1 year of look-back for the rank
    days = px.loc[max(start, pd.Timestamp("2008-01-02")):end].index
    if len(days) == 0:
        return pd.Series(dtype=float)
    weekly = pd.Series(days, index=days).groupby(days.to_period("W")).first()
    vals = {}
    for i, d in enumerate(weekly):
        if progress and i % 10 == 0:
            progress(i / len(weekly), f"Building IV rank for {ticker}: week {i}/{len(weekly)}")
        try:
            vals[d] = _atm_iv(chain(ticker, d, 14, 60))
        except AVError:
            raise
        except Exception:
            vals[d] = np.nan
    iv = pd.Series(vals).sort_index().dropna()
    if iv.empty:
        return pd.Series(dtype=float)
    lo, hi = iv.rolling(52, min_periods=20).min(), iv.rolling(52, min_periods=20).max()
    rank = (100 * (iv - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    return rank.reindex(px.index, method="ffill").loc[days[0]:]


# ----------------------------------------------------------------------------- opening range
def opening_range(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """First regular-hours 30-min bar (9:30–10:00) high/low per day, in the same split-adjusted
    units as daily_prices()."""
    ticker = ticker.upper()
    px = daily_prices(ticker)
    frames = []
    for m in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        key = f"av:orb:{ticker}:{m}"
        hit = store.get(key)
        current = m == pd.Period(_today(), freq="M")
        if hit is not None and (not current or hit[1].date() >= _today()):
            frames.append(hit[0])
            continue
        js = _get("TIME_SERIES_INTRADAY", symbol=ticker, interval="30min", month=str(m),
                  outputsize="full", adjusted="false", extended_hours="false")
        ts = js.get("Time Series (30min)") or {}
        bars = pd.DataFrame.from_dict(ts, orient="index")
        if bars.empty:
            df = pd.DataFrame({"date": [], "orh": [], "orl": []})
        else:
            bars.index = pd.to_datetime(bars.index)
            bars = bars.sort_index()
            first = bars.groupby(bars.index.normalize()).head(1)
            df = pd.DataFrame({"date": first.index.normalize(),
                               "orh": first["2. high"].astype(float).values,
                               "orl": first["3. low"].astype(float).values})
        store.put(key, df)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["orh", "orl"])
    orb = pd.concat(frames).assign(date=lambda d: pd.to_datetime(d["date"])).drop_duplicates("date")
    orb = orb.set_index("date").sort_index()
    ratio = (px["close"] / px["close_raw"]).reindex(orb.index)
    return pd.DataFrame({"orh": orb["orh"] * ratio, "orl": orb["orl"] * ratio})
