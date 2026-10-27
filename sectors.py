"""S&P 500 constituents grouped by GICS sector (from Wikipedia's maintained list)."""
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
