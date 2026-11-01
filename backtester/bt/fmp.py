"""Financial Modeling Prep (FMP Ultimate): daily stock prices + earnings dates."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from . import store
from .config import secret

BASE = "https://financialmodelingprep.com"
_session = requests.Session()


class FMPError(RuntimeError):
    pass


def _get(path: str, **params):
    key = secret("FMP_API_KEY")
    if not key:
        raise FMPError("FMP_API_KEY is missing from Streamlit Secrets.")
    params["apikey"] = key
    r = _session.get(f"{BASE}{path}", params=params, timeout=60)
    if r.status_code in (401, 403):
        raise FMPError(f"FMP refused the request ({r.status_code}). Check FMP_API_KEY / plan.")
    r.raise_for_status()
    return r.json()


def _fresh(fetched_at: dt.datetime) -> bool:
    """Daily-updating data is re-downloaded at most once per day."""
    return fetched_at.date() >= dt.datetime.utcnow().date()


def daily_prices(ticker: str) -> pd.DataFrame:
    """Split-adjusted daily OHLCV, index = date. Full history (FMP Ultimate: ~30 yrs)."""
    ticker = ticker.upper()
    key = f"fmp:px:{ticker}"
    hit = store.get(key)
    if hit is not None and _fresh(hit[1]):
        return _shape_px(hit[0])

    rows = None
    try:
        rows = _get("/stable/historical-price-eod/full", symbol=ticker, **{"from": "1990-01-01"})
    except FMPError:
        raise
    except Exception:
        rows = None
    if not rows:
        js = _get(f"/api/v3/historical-price-full/{ticker}", **{"from": "1990-01-01"})
        rows = js.get("historical", []) if isinstance(js, dict) else []
    if not rows:
        if hit is not None:
            return _shape_px(hit[0])
        raise FMPError(f"FMP returned no price history for {ticker}.")

    df = pd.DataFrame(rows)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    store.put(key, df)
    return _shape_px(df)


def _shape_px(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.drop_duplicates("date").set_index("date").sort_index()
    return out[["open", "high", "low", "close", "volume"]].astype(float)


def earnings_dates(ticker: str) -> list[pd.Timestamp]:
    ticker = ticker.upper()
    key = f"fmp:earn:{ticker}"
    hit = store.get(key)
    if hit is not None and _fresh(hit[1]):
        return _shape_earn(hit[0])
    rows = None
    try:
        rows = _get("/stable/earnings", symbol=ticker, limit=1000)
    except FMPError:
        raise
    except Exception:
        rows = None
    if not rows:
        try:
            rows = _get(f"/api/v3/historical/earning_calendar/{ticker}", limit=1000)
        except Exception:
            rows = []
    df = pd.DataFrame(rows or [], columns=None)
    df = df[["date"]] if "date" in df.columns else pd.DataFrame({"date": []})
    store.put(key, df)
    return _shape_earn(df)


def _shape_earn(df: pd.DataFrame) -> list[pd.Timestamp]:
    if df.empty:
        return []
    return sorted(set(pd.to_datetime(df["date"]).dt.normalize()))
