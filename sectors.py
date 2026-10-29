"""Index member lists: S&P 500 and Nasdaq-100 (Wikipedia), Russell 2000 top holdings (IWM)."""
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500() -> pd.DataFrame:
    """Ticker, Company, Sector, Sub-industry for every S&P 500 member."""
    # Wikipedia refuses requests without a browser-like User-Agent.
    html = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (trading dashboard)"},
                        timeout=30).text
    df = pd.read_html(StringIO(html))[0]
    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["Ticker", "Company", "Sector", "Sub-industry"]
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False).str.strip()  # BRK.B -> BRK-B
    return df.sort_values(["Sector", "Ticker"]).reset_index(drop=True)


NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
IWM_CSV = ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax"
           "?fileType=csv&fileName=IWM_holdings&dataType=fund")
UA = {"User-Agent": "Mozilla/5.0 (trading dashboard)"}
COLUMNS = ["Ticker", "Company", "Sector", "Sub-industry"]


def _clean(tickers: pd.Series) -> pd.Series:
    return tickers.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)


def parse_nasdaq100(html: str) -> pd.DataFrame:
    """The components table from Wikipedia's Nasdaq-100 page."""
    for t in pd.read_html(StringIO(html)):
        cols = {str(c).strip(): c for c in t.columns}
        tcol = next((cols[c] for c in ("Ticker", "Symbol") if c in cols), None)
        if tcol is None or not 90 <= len(t) <= 110:
            continue
        name = next((cols[c] for c in ("Company", "Security", "Name") if c in cols), tcol)
        sector = next((cols[c] for c in ("GICS Sector", "ICB Industry", "Sector") if c in cols), None)
        sub = next((cols[c] for c in ("GICS Sub-Industry", "ICB Subsector", "Industry") if c in cols), None)
        out = pd.DataFrame({"Ticker": _clean(t[tcol]), "Company": t[name].astype(str),
                            "Sector": t[sector].astype(str) if sector is not None else "Nasdaq-100",
                            "Sub-industry": t[sub].astype(str) if sub is not None else "—"})
        return out.sort_values(["Sector", "Ticker"]).reset_index(drop=True)
    raise ValueError("Nasdaq-100 table not found")


def nasdaq100() -> pd.DataFrame:
    """Nasdaq-100 members with sector (from Wikipedia's maintained list)."""
    return parse_nasdaq100(requests.get(NDX_URL, headers=UA, timeout=30).text)


def parse_ishares_csv(text: str) -> pd.DataFrame:
    """iShares holdings download: a few lines of fund details, then the holdings table."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lstrip('﻿"').startswith("Ticker"))
    df = pd.read_csv(StringIO("\n".join(lines[start:])), on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    if "Asset Class" in df:
        df = df[df["Asset Class"].astype(str).str.strip() == "Equity"]
    df = df[df["Ticker"].astype(str).str.fullmatch(r"[A-Za-z][A-Za-z.\-]{0,6}")]
    w = pd.to_numeric(df.get("Weight (%)", pd.Series(dtype=float)).astype(str).str.replace(",", ""),
                      errors="coerce")
    return pd.DataFrame({"Ticker": _clean(df["Ticker"]), "Company": df.get("Name", df["Ticker"]).astype(str),
                         "Sector": df.get("Sector", "Russell 2000").astype(str), "Sub-industry": "—",
                         "Weight": w.values})


def parse_uw_holdings(payload) -> pd.DataFrame:
    """Unusual Whales ETF holdings, whatever the exact field names are."""
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    df = pd.json_normalize(rows) if rows else pd.DataFrame()
    if df.empty:
        raise ValueError("no holdings")
    pick = lambda *names: next((c for c in names if c in df.columns), None)   # noqa: E731
    tcol = pick("ticker", "symbol", "holding_ticker")
    if tcol is None:
        raise ValueError(f"unexpected holdings format: {list(df.columns)[:8]}")
    wcol = pick("weight", "weight_percent", "percent_of_fund", "pct")
    ncol, scol = pick("name", "full_name", "company_name"), pick("sector", "gics_sector")
    return pd.DataFrame({"Ticker": _clean(df[tcol]),
                         "Company": (df[ncol].fillna(df[tcol]) if ncol else df[tcol]).astype(str),
                         "Sector": df[scol].fillna("Other").astype(str) if scol else "Russell 2000",
                         "Sub-industry": "—",
                         "Weight": pd.to_numeric(df[wcol], errors="coerce") if wcol else float("nan")})


def russell2000(top: int = 100) -> tuple[pd.DataFrame, str]:
    """The biggest Russell 2000 companies by weight in IWM, and where the list came from.

    Unusual Whales first (if your plan includes ETF holdings), then iShares' public download.
    """
    errors = []
    try:
        import uw
        if uw.api_key():
            df, src = parse_uw_holdings(uw.get("/api/etfs/IWM/holdings")), "Unusual Whales (IWM holdings)"
        else:
            raise ValueError("no UW key")
    except Exception as e:  # noqa: BLE001
        errors.append(f"UW: {e}")
        try:
            df, src = parse_ishares_csv(requests.get(IWM_CSV, headers=UA, timeout=30).text), "iShares (IWM holdings)"
        except Exception as e2:  # noqa: BLE001
            errors.append(f"iShares: {e2}")
            raise ValueError("Couldn't load the Russell 2000 list – " + "; ".join(errors)) from e2
    df = df[df["Ticker"].str.len() > 0].drop_duplicates("Ticker")
    if df["Weight"].notna().any():
        df = df.sort_values("Weight", ascending=False)
    return df.head(top).reset_index(drop=True), src
