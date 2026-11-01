"""Polygon.io (optional): 30-minute opening range (9:30–10:00 ET) high/low per day."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from . import store
from .config import secret

BASE = "https://api.polygon.io"


def opening_range(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """DataFrame indexed by date with columns orh, orl. Empty if no POLYGON_API_KEY."""
    key_api = secret("POLYGON_API_KEY")
    if not key_api:
        return pd.DataFrame(columns=["orh", "orl"])
    ticker = ticker.upper()
    key = f"poly:orb:{ticker}"
    hit = store.get(key)
    if hit is not None and hit[1].date() >= dt.datetime.utcnow().date():
        return _shape(hit[0])

    frames, url = [], (f"{BASE}/v2/aggs/ticker/{ticker}/range/30/minute/"
                       f"{pd.Timestamp(start).date()}/{pd.Timestamp(end).date()}")
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key_api}
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        frames.append(pd.DataFrame(js.get("results", [])))
        url = js.get("next_url")
        params = {"apiKey": key_api}
    bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if bars.empty:
        df = pd.DataFrame({"date": [], "orh": [], "orl": []})
    else:
        ts = pd.to_datetime(bars["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        first = bars[(ts.dt.hour == 9) & (ts.dt.minute == 30)].copy()
        first["date"] = ts[first.index].dt.tz_localize(None).dt.normalize()
        df = first.rename(columns={"h": "orh", "l": "orl"})[["date", "orh", "orl"]]
    store.put(key, df)
    return _shape(df)


def _shape(df):
    if df.empty:
        return pd.DataFrame(columns=["orh", "orl"])
    return df.assign(date=pd.to_datetime(df["date"])).set_index("date")[["orh", "orl"]].sort_index()
