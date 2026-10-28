"""Market data layer.

Live data comes from yfinance (free, ~15 min delay on some feeds, no key).
If a live call fails (no internet, rate limit, Yahoo outage) every function
falls back to deterministic synthetic data so the dashboard still renders.
Every fallback is recorded in WARNINGS so the UI can tell you.

To swap in a paid feed (Polygon, Alpaca, IBKR, Schwab...), reimplement the
functions in this file with the same return shapes. Nothing else changes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

ET = ZoneInfo("America/New_York")
WARNINGS: list[str] = []
FORCE_DEMO = False  # set by the app sidebar; plugins inherit it

MACRO = {
    "ES=F": "S&P 500 fut",
    "NQ=F": "Nasdaq fut",
    "YM=F": "Dow fut",
    "RTY=F": "Russell fut",
    "^VIX": "VIX",
    "^TNX": "10Y yield",
    "DX-Y.NYB": "Dollar idx",
    "CL=F": "Crude oil",
    "GC=F": "Gold",
    "BTC-USD": "Bitcoin",
}


# Digital commodities named in the SEC/CFTC release, with their Yahoo Finance symbols.
CRYPTO = {
    "BTC": ("Bitcoin", "BTC-USD"), "ETH": ("Ether", "ETH-USD"), "SOL": ("Solana", "SOL-USD"),
    "XRP": ("XRP", "XRP-USD"), "ADA": ("Cardano", "ADA-USD"), "AVAX": ("Avalanche", "AVAX-USD"),
    "LINK": ("Chainlink", "LINK-USD"), "DOGE": ("Dogecoin", "DOGE-USD"), "DOT": ("Polkadot", "DOT-USD"),
    "LTC": ("Litecoin", "LTC-USD"), "BCH": ("Bitcoin Cash", "BCH-USD"), "HBAR": ("Hedera", "HBAR-USD"),
    "XLM": ("Stellar", "XLM-USD"), "XTZ": ("Tezos", "XTZ-USD"), "SHIB": ("Shiba Inu", "SHIB-USD"),
    "APT": ("Aptos", "APT21794-USD"), "ALGO": ("Algorand", "ALGO-USD"),
    "LBC": ("LBRY Credits", "LBC-USD"),
}


def read_watchlist(text: str) -> tuple[list[str], list[str]]:
    """Split the master watchlist into (stocks, crypto). '#' lines are notes;
    everything after a '# CRYPTO' heading is crypto."""
    stocks: list[str] = []
    crypto: list[str] = []
    section = stocks
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            if line.upper().startswith("# CRYPTO"):
                section = crypto
            continue
        section.extend(t.strip().upper() for t in line.replace(",", " ").split() if t.strip())
    return list(dict.fromkeys(stocks)), list(dict.fromkeys(crypto))


def crypto_snapshot(symbols: list[str], demo: bool = False) -> pd.DataFrame:
    """Price and 24h / 7d / 30d change for each coin (crypto trades 24/7: no pre-market)."""
    rows = []
    for sym in symbols:
        name, yahoo = CRYPTO.get(sym, (sym, f"{sym}-USD"))
        d = daily_history(yahoo, period="3mo", demo=demo, fallback_demo=False)
        c = d["Close"].dropna()
        if c.empty:   # never show made-up prices for a coin with no real data
            rows.append({"Coin": sym, "Name": name, "Price": np.nan, "24h %": np.nan,
                         "7d %": np.nan, "30d %": np.nan})
            continue

        def chg(n):
            return (c.iloc[-1] / c.iloc[-1 - n] - 1) * 100 if len(c) > n else np.nan

        rows.append({"Coin": sym, "Name": name, "Price": float(c.iloc[-1]),
                     "24h %": chg(1), "7d %": chg(7), "30d %": chg(30)})
    return pd.DataFrame(rows)


# Funds never report earnings; skip them in demo mode.
ETFS = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
        "XLU", "XLB", "XLRE", "XLC", "SMH", "SOXX", "ARKK", "TLT", "GLD", "SLV"}


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def session_status(t: dt.datetime | None = None) -> str:
    t = t or now_et()
    if t.weekday() >= 5:
        return "Closed (weekend)"
    hm = t.hour * 60 + t.minute
    if 4 * 60 <= hm < 9 * 60 + 30:
        return "Pre-market"
    if 9 * 60 + 30 <= hm < 16 * 60:
        return "Regular session"
    if 16 * 60 <= hm < 20 * 60:
        return "After-hours"
    return "Closed"


def _warn(msg: str) -> None:
    if msg not in WARNINGS:
        WARNINGS.append(msg)


# --------------------------------------------------------------------------- #
# Synthetic data (demo mode / fallback)
# --------------------------------------------------------------------------- #
_BASE = {"SPY": 580, "QQQ": 500, "IWM": 225, "XLK": 235, "XLF": 48, "XLE": 90,
         "AAPL": 235, "MSFT": 430, "NVDA": 140, "AMZN": 190, "META": 580,
         "GOOGL": 170, "TSLA": 250, "AMD": 150, "AVGO": 175, "NFLX": 760,
         "ES=F": 5850, "NQ=F": 20600, "YM=F": 42500, "RTY=F": 2250, "^VIX": 17,
         "^TNX": 4.2, "DX-Y.NYB": 104, "CL=F": 71, "GC=F": 2700, "BTC-USD": 68000}


def _rng(ticker: str, salt: str = "") -> np.random.Generator:
    seed = int(hashlib.md5(f"{ticker}{salt}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _base_price(ticker: str) -> float:
    return _BASE.get(ticker, float(_rng(ticker).uniform(20, 400)))


def _demo_daily(ticker: str, days: int = 1260) -> pd.DataFrame:
    rng = _rng(ticker, "daily")
    idx = pd.bdate_range(end=now_et().date(), periods=days)
    vol = rng.uniform(0.012, 0.03)
    rets = rng.normal(0.0004, vol, days)
    close = _base_price(ticker) * np.exp(np.cumsum(rets) - rets.sum())
    open_ = close * np.exp(rng.normal(0, vol / 3, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, days)))
    volume = rng.lognormal(16, 0.4, days).astype(int)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


def _demo_intraday(ticker: str, days: int = 5) -> pd.DataFrame:
    rng = _rng(ticker, "intraday")
    frames = []
    price = _base_price(ticker)
    for d in pd.bdate_range(end=now_et().date(), periods=days):
        times = pd.date_range(dt.datetime.combine(d, dt.time(4, 0)),
                              dt.datetime.combine(d, dt.time(19, 55)),
                              freq="5min", tz=ET)
        reg = (times.time >= dt.time(9, 30)) & (times.time < dt.time(16, 0))
        sig = np.where(reg, 0.0012, 0.0006)
        rets = rng.normal(0, sig)
        close = price * np.exp(np.cumsum(rets))
        open_ = np.r_[price, close[:-1]]
        spread = np.abs(rng.normal(0, sig, len(times))) * close
        vol = np.where(reg, rng.lognormal(11, 0.5, len(times)),
                       rng.lognormal(7.5, 0.8, len(times))).astype(int)
        frames.append(pd.DataFrame({
            "Open": open_, "High": np.maximum(open_, close) + spread,
            "Low": np.minimum(open_, close) - spread, "Close": close,
            "Volume": vol}, index=times))
        price = close[-1]
    df = pd.concat(frames)
    return df[df.index <= now_et()]  # today's bars stop at "now", like a live feed


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _uw_candles(ticker: str, size: str) -> pd.DataFrame | None:
    """Unusual Whales candles when a key is configured (real-time); None otherwise."""
    import uw
    if not uw.api_key():
        return None
    try:
        df = uw.candles(ticker, size)
        return df if len(df) else None
    except Exception as e:  # noqa: BLE001
        _warn(f"{ticker} UW {size}: {e}")
        return None


def _empty(tz) -> pd.DataFrame:
    """No candles – but with a time index, so time-of-day code still works on it."""
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"], dtype=float,
                        index=pd.DatetimeIndex([], tz=tz))


def _live(demo: bool) -> bool:
    return not (demo or FORCE_DEMO)


def daily_history(ticker: str, period: str = "5y", demo: bool = False,
                  fallback_demo: bool = False) -> pd.DataFrame:
    """Daily candles: Yahoo first, then Unusual Whales. In live mode a ticker with no real
    data returns an EMPTY frame – never made-up numbers. Demo candles are only used when
    Demo data is switched on."""
    if _live(demo):
        if yf is not None:
            try:
                df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
                if not df.empty:
                    df.index = df.index.tz_localize(None)
                    return df[["Open", "High", "Low", "Close", "Volume"]]
            except Exception as e:  # noqa: BLE001
                _warn(f"{ticker} daily: {e.__class__.__name__}")
        uwd = _uw_candles(ticker, "1d")
        if uwd is not None:
            uwd.index = uwd.index.tz_localize(None).normalize()
            return uwd[~uwd.index.duplicated(keep="last")]
        if not fallback_demo:
            _warn(f"{ticker}: no data – check the symbol")
            return _empty(None)
    return _demo_daily(ticker)


def intraday(ticker: str, days: int = 5, interval: str = "5m",
             demo: bool = False) -> pd.DataFrame:
    """Intraday bars INCLUDING pre-market and after-hours, indexed in ET.

    Unusual Whales (real-time) is used first for recent days when a key is set;
    Yahoo covers longer look-backs (up to 60 days of 5m bars).
    """
    if not (demo or FORCE_DEMO) and days <= 5:
        uwd = _uw_candles(ticker, interval)
        if uwd is not None:
            return uwd
    if not (demo or FORCE_DEMO) and yf is not None:
        try:
            df = yf.Ticker(ticker).history(period=f"{days}d", interval=interval,
                                           prepost=True, auto_adjust=False)
            if not df.empty:
                df.index = df.index.tz_convert(ET)
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:  # noqa: BLE001
            _warn(f"{ticker} intraday: {e.__class__.__name__}")
        uwd = _uw_candles(ticker, interval) if days > 5 else None
        if uwd is not None:
            return uwd
    if _live(demo):
        _warn(f"{ticker}: no intraday data – check the symbol")
        return _empty(ET)
    return _demo_intraday(ticker, days)


def premarket_snapshot(tickers: list[str], demo: bool = False) -> pd.DataFrame:
    """One row per ticker: prior close, latest pre-market print, gap, PM range/volume.

    Robust to thin ETFs (e.g. XLK) that may have *no* pre-market trades yet:
    the row is kept and pm_last is NaN instead of the ticker disappearing.
    """
    rows = []
    for t in tickers:
        df = intraday(t, days=5, interval="5m", demo=demo)
        row = {"Ticker": t}
        if df.empty:
            rows.append(row)
            continue
        dates = sorted(set(df.index.date))
        today = dates[-1]
        tod = df.index.time
        regular = (tod >= dt.time(9, 30)) & (tod < dt.time(16, 0))
        prev_reg = df[regular & (df.index.date < today)]
        today_df = df[df.index.date == today]
        pm = today_df[today_df.index.time < dt.time(9, 30)]
        pm = pm[pm["Volume"] > 0] if (pm["Volume"] > 0).any() else pm
        prev_close = prev_reg["Close"].iloc[-1] if len(prev_reg) else np.nan
        row["Prev close"] = prev_close
        if len(pm):
            last = pm["Close"].iloc[-1]
            row.update({"PM last": last, "PM high": pm["High"].max(),
                        "PM low": pm["Low"].min(), "PM volume": int(pm["Volume"].sum()),
                        "Last PM print": pm.index[-1].strftime("%H:%M")})
        else:
            # Fallback for ETFs without PM bars: Yahoo's quote field, if present.
            last = np.nan
            if not (demo or FORCE_DEMO) and yf is not None:
                try:
                    last = yf.Ticker(t).info.get("preMarketPrice") or np.nan
                except Exception:  # noqa: BLE001
                    pass
            row.update({"PM last": last, "PM volume": 0,
                        "Last PM print": "no PM trades yet"})
        row["Gap %"] = (last / prev_close - 1) * 100 if prev_close else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def macro_snapshot(demo: bool = False) -> pd.DataFrame:
    rows = []
    for sym, name in MACRO.items():
        d = daily_history(sym, period="5d", demo=demo)
        i = intraday(sym, days=2, interval="5m", demo=demo)
        if d.empty and i.empty:
            continue
        last = i["Close"].iloc[-1] if len(i) else d["Close"].iloc[-1]
        if d.empty:
            prev = np.nan
        else:
            # If today's daily row already exists, prior close is the row before it.
            has_today = d.index[-1].date() == now_et().date()
            prev = d["Close"].iloc[-2] if has_today and len(d) > 1 else d["Close"].iloc[-1]
        rows.append({"Symbol": sym, "Name": name, "Last": last,
                     "Chg %": (last / prev - 1) * 100 if prev else np.nan})
    return pd.DataFrame(rows, columns=["Symbol", "Name", "Last", "Chg %"])


def next_earnings(ticker: str, demo: bool = False) -> dict:
    if not (demo or FORCE_DEMO) and yf is not None:
        try:
            cal = yf.Ticker(ticker).calendar or {}
            dates = cal.get("Earnings Date") or []
            if dates:
                return {"Ticker": ticker, "Date": pd.Timestamp(dates[0]).date(),
                        "EPS est": cal.get("Earnings Average"),
                        "Revenue est": cal.get("Revenue Average")}
            return {"Ticker": ticker, "Date": None}
        except Exception as e:  # noqa: BLE001
            _warn(f"{ticker} earnings: {e.__class__.__name__}")
    if _live(demo) or ticker in ETFS:
        return {"Ticker": ticker, "Date": None}  # ETFs don't report
    rng = _rng(ticker, "earn")
    d = now_et().date() + dt.timedelta(days=int(rng.integers(1, 60)))
    return {"Ticker": ticker, "Date": d, "EPS est": round(float(rng.uniform(0.3, 5)), 2),
            "Revenue est": None}


def _session_of(ts: pd.Timestamp) -> str:
    if ts.hour == 0 and ts.minute == 0:
        return "Time not confirmed"
    hm = ts.hour * 60 + ts.minute
    return ("Before open (BMO)" if hm < 570 else "After close (AMC)" if hm >= 960
            else "During market hours")


def earnings_between(tickers: list[str], start: dt.date, end: dt.date,
                     demo: bool = False) -> pd.DataFrame:
    """Watchlist earnings reports between two dates, with time of day and BMO/AMC."""
    rows = []
    for t in tickers:
        if not (demo or FORCE_DEMO) and yf is not None:
            try:
                ed = yf.Ticker(t).get_earnings_dates(limit=8)
            except Exception as e:  # noqa: BLE001
                _warn(f"{t} earnings dates: {e.__class__.__name__}")
                continue
            if ed is None or ed.empty:
                continue
            for ts, r in ed.iterrows():
                ts = ts.tz_convert(ET)
                if start <= ts.date() <= end:
                    rows.append({"Date": ts.date(), "Day": f"{ts:%a}", "Ticker": t,
                                 "Time (ET)": "—" if ts.hour == 0 and ts.minute == 0 else f"{ts:%I:%M %p}",
                                 "When": _session_of(ts), "EPS est": r.get("EPS Estimate"),
                                 "EPS actual": r.get("Reported EPS")})
        elif t not in ETFS:
            rng = _rng(t, f"week{start}")
            if rng.random() < 0.3:
                d = start + dt.timedelta(days=int(rng.integers(0, 5)))
                ts = pd.Timestamp(dt.datetime.combine(d, dt.time(16, 5) if rng.random() < 0.6
                                                      else dt.time(7, 0)), tz=ET)
                rows.append({"Date": d, "Day": f"{ts:%a}", "Ticker": t, "Time (ET)": f"{ts:%I:%M %p}",
                             "When": _session_of(ts), "EPS est": round(float(rng.uniform(0.2, 4)), 2),
                             "EPS actual": None})
    out = pd.DataFrame(rows, columns=["Date", "Day", "Ticker", "Time (ET)", "When", "EPS est", "EPS actual"])
    return out.sort_values(["Date", "When", "Ticker"]).reset_index(drop=True)


def past_earnings_dates(ticker: str, demo: bool = False) -> list[pd.Timestamp]:
    """Past report timestamps (with time of day when known)."""
    if not (demo or FORCE_DEMO) and yf is not None:
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=16)
            if ed is not None and len(ed):
                past = [ts.tz_convert(ET) for ts in ed.index if ts.tz_convert(ET) < now_et()]
                return sorted(past)
        except Exception as e:  # noqa: BLE001
            _warn(f"{ticker} earnings history: {e.__class__.__name__}")
    if _live(demo):
        return []
    rng = _rng(ticker, "pastearn")
    out, d = [], now_et() - dt.timedelta(days=int(rng.integers(20, 80)))
    for _ in range(12):
        hour = 16 if rng.random() < 0.6 else 7
        out.append(pd.Timestamp(d.date()).tz_localize(ET) + pd.Timedelta(hours=hour))
        d -= dt.timedelta(days=91)
    return sorted(out)


def expirations(ticker: str, demo: bool = False) -> list[str]:
    if not (demo or FORCE_DEMO) and yf is not None:
        try:
            exps = list(yf.Ticker(ticker).options)
            if exps:
                return exps
        except Exception as e:  # noqa: BLE001
            _warn(f"{ticker} options: {e.__class__.__name__}")
    if _live(demo):
        _warn(f"{ticker}: no option chain available")
        return []
    today = now_et().date()
    fridays = [today + dt.timedelta(days=(4 - today.weekday()) % 7 + 7 * k) for k in range(8)]
    leaps = [dt.date(today.year + y, 1, 15) for y in (1, 2, 3)]
    leaps = [d - dt.timedelta(days=(d.weekday() - 4) % 7) for d in leaps]
    monthly = [dt.date(today.year + (today.month + m - 1) // 12, (today.month + m - 1) % 12 + 1, 15)
               for m in (1, 2, 3, 6, 9)]
    return sorted({d.isoformat() for d in fridays + monthly + leaps})


def option_chain(ticker: str, expiry: str, spot: float, demo: bool = False) -> pd.DataFrame:
    """Calls + puts for one expiry: strike, bid, ask, mid, iv, volume, oi, type."""
    if not (demo or FORCE_DEMO) and yf is not None:
        try:
            ch = yf.Ticker(ticker).option_chain(expiry)
            frames = []
            for kind, df in (("call", ch.calls), ("put", ch.puts)):
                df = df.rename(columns={"impliedVolatility": "iv", "openInterest": "oi",
                                        "volume": "volume", "lastPrice": "last"})
                df["type"] = kind
                frames.append(df[["strike", "bid", "ask", "last", "iv", "volume", "oi", "type"]])
            out = pd.concat(frames, ignore_index=True)
            out["mid"] = np.where((out.bid > 0) & (out.ask > 0), (out.bid + out.ask) / 2, out["last"])
            out["expiry"] = expiry
            return out
        except Exception as e:  # noqa: BLE001
            _warn(f"{ticker} chain {expiry}: {e.__class__.__name__}")
    if _live(demo):
        return pd.DataFrame(columns=["strike", "bid", "ask", "last", "iv", "volume", "oi", "type",
                                     "mid", "expiry"])
    from analytics import bs_price  # local import avoids a cycle
    rng = _rng(ticker, expiry)
    T = max((pd.Timestamp(expiry).date() - now_et().date()).days, 1) / 365
    base_iv = rng.uniform(0.22, 0.5)
    strikes = np.round(np.linspace(spot * 0.5, spot * 1.5, 41), 0 if spot > 50 else 1)
    rows = []
    for k in strikes:
        iv = base_iv * (1 + 0.25 * (np.log(spot / k)) ** 2 + 0.08 * np.log(spot / k))
        for kind in ("call", "put"):
            mid = bs_price(spot, k, T, 0.04, iv, kind)
            spr = max(0.02, mid * 0.02)
            rows.append({"strike": k, "bid": max(mid - spr / 2, 0), "ask": mid + spr / 2,
                         "last": mid, "mid": mid, "iv": iv, "type": kind,
                         "volume": int(rng.integers(0, 2000)), "oi": int(rng.integers(10, 20000))})
    out = pd.DataFrame(rows)
    out["expiry"] = expiry
    return out
