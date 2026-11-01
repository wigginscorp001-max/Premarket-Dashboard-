"""ORATS Data API: historical option chains (bid/ask/greeks) and IV rank history.

Endpoints (https://api.orats.io/datav2):
  hist/strikes  ?ticker=&tradeDate=YYYY-MM-DD&dte=min,max   -> one row per strike & expiry
  hist/ivrank   ?ticker=                                    -> daily IV rank history
  hist/summaries?ticker=                                    -> fallback (iv30d) to compute IV rank

ORATS snapshots are taken shortly before the close each trading day; the engine therefore
only trades on snapshots of the day AFTER a signal (no look-ahead).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from . import store
from .config import secret

BASE = "https://api.orats.io/datav2"
_session = requests.Session()

CHAIN_COLS = ["expiry", "dte", "strike", "spot", "type", "bid", "ask", "delta", "oi", "iv"]


class ORATSError(RuntimeError):
    pass


def _get(path: str, **params) -> list[dict]:
    token = secret("ORATS_API_KEY")
    if not token:
        raise ORATSError("ORATS_API_KEY is missing from Streamlit Secrets.")
    params["token"] = token
    for attempt in range(5):
        r = _session.get(f"{BASE}/{path}", params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (401, 403):
            raise ORATSError(f"ORATS refused the request ({r.status_code}). Check ORATS_API_KEY / plan.")
        if r.status_code == 404:
            return []
        r.raise_for_status()
        js = r.json()
        return js.get("data", []) if isinstance(js, dict) else (js or [])
    raise ORATSError(f"ORATS kept failing for {path} {params.get('ticker')} {params.get('tradeDate')}")


def chain(ticker: str, date: pd.Timestamp, dte_min: int, dte_max: int) -> pd.DataFrame:
    """Normalised chain for one trade date: one row per contract (calls and puts)."""
    ticker = ticker.upper()
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    key = f"orats:chain:{ticker}:{d}:{dte_min}-{dte_max}"
    hit = store.get(key)
    if hit is not None:            # history never changes -> cache forever
        return hit[0]
    rows = _get("hist/strikes", ticker=ticker, tradeDate=d, dte=f"{dte_min},{dte_max}")
    df = normalise(pd.DataFrame(rows), dte_min, dte_max)
    store.put(key, df)
    return df


def normalise(raw: pd.DataFrame, dte_min: int = 0, dte_max: int = 5000) -> pd.DataFrame:
    if raw.empty or "strike" not in raw.columns:
        return pd.DataFrame(columns=CHAIN_COLS)
    raw = raw.copy()
    raw["expiry"] = pd.to_datetime(raw["expirDate"]).dt.normalize()
    if "dte" not in raw.columns:
        raw["dte"] = (raw["expiry"] - pd.to_datetime(raw["tradeDate"])).dt.days
    raw = raw[(raw["dte"] >= dte_min) & (raw["dte"] <= dte_max)]
    g = lambda c: raw[c] if c in raw.columns else np.nan
    iv = g("smvVol")
    base = pd.DataFrame({"expiry": raw["expiry"], "dte": raw["dte"].astype(int),
                         "strike": raw["strike"].astype(float), "spot": g("stockPrice")})
    calls = base.assign(type="C", bid=g("callBidPrice"), ask=g("callAskPrice"),
                        delta=g("delta"), oi=g("callOpenInterest"), iv=g("callMidIv").fillna(iv)
                        if "callMidIv" in raw.columns else iv)
    puts = base.assign(type="P", bid=g("putBidPrice"), ask=g("putAskPrice"),
                       delta=g("delta") - 1.0, oi=g("putOpenInterest"), iv=g("putMidIv").fillna(iv)
                       if "putMidIv" in raw.columns else iv)
    out = pd.concat([calls, puts], ignore_index=True)[CHAIN_COLS]
    for c in ["spot", "bid", "ask", "delta", "oi", "iv"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def iv_rank(ticker: str) -> pd.Series:
    """Daily 1-year IV rank (0–100), index = date."""
    import datetime as dt
    ticker = ticker.upper()
    key = f"orats:ivr:{ticker}"
    hit = store.get(key)
    if hit is not None and hit[1].date() >= dt.datetime.utcnow().date():
        return _shape_ivr(hit[0])

    df = pd.DataFrame()
    try:
        rows = _get("hist/ivrank", ticker=ticker)
        df = pd.DataFrame(rows)
        if not df.empty and "ivRank1y" in df.columns:
            df = pd.DataFrame({"date": df["tradeDate"], "ivr": df["ivRank1y"]})
        else:
            df = pd.DataFrame()
    except ORATSError:
        raise
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        rows = _get("hist/summaries", ticker=ticker)
        s = pd.DataFrame(rows)
        if not s.empty and "iv30d" in s.columns:
            s = s.sort_values("tradeDate")
            iv = s["iv30d"].astype(float)
            lo, hi = iv.rolling(252, min_periods=60).min(), iv.rolling(252, min_periods=60).max()
            df = pd.DataFrame({"date": s["tradeDate"], "ivr": 100 * (iv - lo) / (hi - lo)})
    if df.empty:
        df = pd.DataFrame({"date": [], "ivr": []})
    store.put(key, df)
    return _shape_ivr(df)


def _shape_ivr(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(df["ivr"], errors="coerce").values,
                  index=pd.to_datetime(df["date"]).dt.normalize())
    return s[~s.index.duplicated()].sort_index()
