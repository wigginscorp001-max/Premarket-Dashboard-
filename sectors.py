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
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [c[-1] for c in t.columns]
        cols = {str(c).strip().split("[")[0].strip(): c for c in t.columns}
        tcol = next((cols[c] for c in ("Ticker", "Symbol", "Ticker symbol") if c in cols), None)
        if tcol is None or not 90 <= len(t) <= 120:
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
    """Nasdaq-100 members: Unusual Whales QQQ holdings first, then Wikipedia's list."""
    try:
        import uw
        if uw.api_key():
            df = parse_uw_holdings(uw.get("/api/etfs/QQQ/holdings"))
            df = df[df["Ticker"].str.fullmatch(r"[A-Z][A-Z\-]{0,5}")].drop_duplicates("Ticker")
            if len(df) >= 90:
                return df.drop(columns="Weight").sort_values(["Sector", "Ticker"]).reset_index(drop=True)
    except Exception:  # noqa: BLE001
        pass
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


# Major US-listed ETFs by category (the most traded funds in each group).
ETFS = {
    "Broad market": {"SPY": "S&P 500", "VOO": "Vanguard S&P 500", "IVV": "iShares S&P 500", "QQQ": "Nasdaq-100",
                     "DIA": "Dow 30", "IWM": "Russell 2000", "VTI": "Total US market", "RSP": "S&P 500 equal weight",
                     "MDY": "S&P MidCap 400", "IJR": "S&P SmallCap 600", "VTV": "Value", "VUG": "Growth",
                     "SCHD": "Dividend", "VIG": "Dividend growth", "MTUM": "Momentum", "QUAL": "Quality",
                     "USMV": "Min volatility", "SPLG": "S&P 500 (low cost)", "QQQM": "Nasdaq-100 (low cost)"},
    "Sectors (SPDR)": {"XLK": "Technology", "XLC": "Communication", "XLY": "Consumer discretionary",
                       "XLP": "Consumer staples", "XLE": "Energy", "XLF": "Financials", "XLV": "Health care",
                       "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real estate", "XLU": "Utilities"},
    "Industries & themes": {"SMH": "Semiconductors", "SOXX": "Semiconductors (iShares)", "IGV": "Software",
                            "XBI": "Biotech", "IBB": "Biotech (large)", "KRE": "Regional banks", "KBE": "Banks",
                            "XHB": "Homebuilders", "ITB": "Home construction", "XRT": "Retail", "XOP": "Oil & gas E&P",
                            "OIH": "Oil services", "JETS": "Airlines", "ITA": "Aerospace & defense",
                            "XAR": "Aerospace & defense (equal wt)", "TAN": "Solar", "ICLN": "Clean energy",
                            "LIT": "Lithium & batteries", "URA": "Uranium", "COPX": "Copper miners",
                            "GDX": "Gold miners", "GDXJ": "Junior gold miners", "SIL": "Silver miners",
                            "ARKK": "ARK Innovation", "BOTZ": "Robotics & AI", "CIBR": "Cybersecurity",
                            "SKYY": "Cloud", "HACK": "Cybersecurity (ETFMG)", "PAVE": "Infrastructure",
                            "IYT": "Transportation", "VNQ": "REITs", "XME": "Metals & mining", "MOO": "Agribusiness"},
    "International": {"EFA": "Developed markets", "EEM": "Emerging markets", "VEA": "Developed (Vanguard)",
                      "VWO": "Emerging (Vanguard)", "FXI": "China large-cap", "KWEB": "China internet",
                      "MCHI": "China", "EWJ": "Japan", "EWZ": "Brazil", "INDA": "India", "EWG": "Germany",
                      "EWU": "United Kingdom", "EWT": "Taiwan", "EWY": "South Korea", "EWW": "Mexico",
                      "EWC": "Canada", "EWA": "Australia", "VGK": "Europe", "ACWI": "All-world"},
    "Bonds & rates": {"TLT": "20+ yr Treasuries", "IEF": "7–10 yr Treasuries", "SHY": "1–3 yr Treasuries",
                      "BIL": "T-bills", "SGOV": "0–3 month T-bills", "AGG": "US aggregate bond", "BND": "Total bond",
                      "LQD": "Investment-grade corporate", "HYG": "High yield", "JNK": "High yield (SPDR)",
                      "TIP": "TIPS (inflation)", "EMB": "EM bonds", "MUB": "Municipal", "TBT": "2x short 20+ yr Treasuries"},
    "Commodities & currencies": {"GLD": "Gold", "IAU": "Gold (iShares)", "SLV": "Silver", "USO": "Crude oil",
                                 "UNG": "Natural gas", "DBC": "Commodities basket", "DBA": "Agriculture",
                                 "CPER": "Copper", "PPLT": "Platinum", "UUP": "US dollar", "FXE": "Euro",
                                 "FXY": "Japanese yen"},
    "Volatility": {"VXX": "VIX short-term futures", "UVXY": "1.5x VIX short-term", "SVXY": "-0.5x VIX short-term",
                   "VIXY": "VIX short-term (ProShares)"},
    "Leveraged & inverse": {"TQQQ": "3x Nasdaq-100", "SQQQ": "-3x Nasdaq-100", "SPXL": "3x S&P 500",
                            "SPXS": "-3x S&P 500", "UPRO": "3x S&P 500 (ProShares)", "SPXU": "-3x S&P 500 (ProShares)",
                            "SSO": "2x S&P 500", "SDS": "-2x S&P 500", "SH": "-1x S&P 500", "PSQ": "-1x Nasdaq-100",
                            "TNA": "3x Russell 2000", "TZA": "-3x Russell 2000", "SOXL": "3x Semiconductors",
                            "SOXS": "-3x Semiconductors", "TECL": "3x Technology", "FAS": "3x Financials",
                            "FAZ": "-3x Financials", "LABU": "3x Biotech", "LABD": "-3x Biotech",
                            "NUGT": "2x Gold miners", "DUST": "-2x Gold miners", "TMF": "3x 20+ yr Treasuries",
                            "TMV": "-3x 20+ yr Treasuries", "NVDL": "2x NVIDIA", "TSLL": "2x Tesla",
                            "BITX": "2x Bitcoin"},
    "Crypto": {"IBIT": "Bitcoin (iShares)", "FBTC": "Bitcoin (Fidelity)", "GBTC": "Bitcoin (Grayscale)",
               "BITO": "Bitcoin futures", "ETHA": "Ether (iShares)", "FETH": "Ether (Fidelity)",
               "ETHE": "Ether (Grayscale)", "BITB": "Bitcoin (Bitwise)"},
    "Income & options strategies": {"JEPI": "Equity premium income", "JEPQ": "Nasdaq premium income",
                                    "QYLD": "Nasdaq covered calls", "XYLD": "S&P covered calls",
                                    "RYLD": "Russell covered calls", "DIVO": "Dividend + covered calls"},
}


def etf_list() -> pd.DataFrame:
    """ETFs as Ticker / Company / Sector (category) / Sub-industry, like the S&P 500 list."""
    rows = [{"Ticker": t, "Company": name, "Sector": cat, "Sub-industry": cat}
            for cat, funds in ETFS.items() for t, name in funds.items()]
    return pd.DataFrame(rows).drop_duplicates("Ticker").reset_index(drop=True)
