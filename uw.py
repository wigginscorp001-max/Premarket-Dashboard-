"""Unusual Whales API client (https://unusualwhales.com/public-api).

Your key is read, in order, from:
  1. the UW_API_KEY environment variable
  2. a `.env` file next to this script containing  UW_API_KEY=your_key
  3. ~/TradingJournal/uw_config.json  ({"api_token": "..."})
  4. Streamlit secrets (the hosted website):  UW_API_KEY = "..."
Never put the key in the code.

Endpoint paths follow UW's public API docs. If UW renames one, fix it in
ENDPOINTS below, or use the "Any endpoint" box in the dashboard to try a path.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.unusualwhales.com"
ROOT = Path(__file__).parent

ENDPOINTS = {
    "flow_alerts": "/api/option-trades/flow-alerts",
    "market_tide": "/api/market/market-tide",
    "darkpool_recent": "/api/darkpool/recent",
    "darkpool_ticker": "/api/darkpool/{ticker}",
    "gex_by_strike": "/api/stock/{ticker}/greek-exposure/strike",
    "gex_daily": "/api/stock/{ticker}/greek-exposure",
    "options_volume": "/api/stock/{ticker}/options-volume",
    "iv_rank": "/api/stock/{ticker}/iv-rank",
    "max_pain": "/api/stock/{ticker}/max-pain",
    "earnings_premarket": "/api/earnings/premarket",
    "earnings_afterhours": "/api/earnings/afterhours",
    "earnings_ticker": "/api/earnings/{ticker}",
    "ohlc": "/api/stock/{ticker}/ohlc/{size}",
    "flow_ticker": "/api/stock/{ticker}/flow-recent",
    "net_prem_ticks": "/api/stock/{ticker}/net-prem-ticks",
}


class UWError(Exception):
    pass


def api_key() -> str | None:
    key = os.environ.get("UW_API_KEY")
    if key:
        return key.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("UW_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    try:  # hosted on Streamlit Cloud: key lives in the app's Secrets box
        import streamlit as st
        if "UW_API_KEY" in st.secrets:
            return str(st.secrets["UW_API_KEY"]).strip()
    except Exception:  # noqa: BLE001  (no secrets file locally)
        pass
    # Reuse the token already saved for the TradingJournal scripts.
    cfg = Path.home() / "TradingJournal" / "uw_config.json"
    if cfg.exists():
        import json
        try:
            return json.loads(cfg.read_text()).get("api_token") or None
        except ValueError:
            pass
    return None


def get(path: str, params: dict | None = None, ticker: str | None = None):
    """GET an endpoint (a key of ENDPOINTS or a raw '/api/...' path). Returns parsed JSON."""
    key = api_key()
    if not key:
        raise UWError("No Unusual Whales key found. Add UW_API_KEY to Streamlit Secrets (or .env on your Mac).")
    path = ENDPOINTS.get(path, path)
    if ticker:
        path = path.replace("{ticker}", ticker.upper())
    params = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
    r = requests.get(BASE + path, params=params, timeout=20,
                     headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    if r.status_code == 401:
        raise UWError("Unusual Whales rejected the key (401). Check UW_API_KEY in Streamlit Secrets (or .env on your Mac).")
    if r.status_code == 403:
        raise UWError(f"Your UW plan doesn't include {path} (403).")
    if r.status_code == 404:
        raise UWError(f"Endpoint not found: {path} (404). UW may have renamed it.")
    if r.status_code == 429:
        raise UWError("Rate limited by Unusual Whales (429). Wait a minute and refresh.")
    if not r.ok:
        raise UWError(f"{path}: HTTP {r.status_code} – {r.text[:200]}")
    return r.json()


def to_df(payload) -> pd.DataFrame:
    """UW wraps most results as {"data": [...]}. Flatten to a DataFrame."""
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        rows = [rows]
    df = pd.json_normalize(rows) if rows else pd.DataFrame()
    # Numbers often arrive as strings ("1234.5"); convert where it's clean.
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() and conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    return df


def fetch(name: str, params: dict | None = None, ticker: str | None = None) -> pd.DataFrame:
    return to_df(get(name, params, ticker))


def candles(ticker: str, size: str = "5m") -> pd.DataFrame:
    """OHLCV candles (incl. pre/post market) as Open/High/Low/Close/Volume, ET index."""
    path = ENDPOINTS["ohlc"].replace("{size}", size)
    df = to_df(get(path, ticker=ticker))
    if df.empty:
        return df
    tcol = next((c for c in ("start_time", "timestamp", "time", "date", "end_time")
                 if c in df.columns), None)
    if tcol is None:
        raise UWError(f"Unexpected candle format: {list(df.columns)}")
    idx = pd.to_datetime(df[tcol], utc=True).dt.tz_convert("America/New_York")
    out = pd.DataFrame({c.title(): pd.to_numeric(df[c], errors="coerce")
                        for c in ("open", "high", "low", "close", "volume") if c in df.columns})
    out.index = pd.DatetimeIndex(idx)
    return out.sort_index()[~out.index.duplicated()]
