"""Indicators, setup scanners and probability math.

All probabilities are statistical estimates (lognormal / historical frequency),
not predictions. Use them to rank and size ideas, not as guarantees.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def realized_vol(close: pd.Series, n: int = 20) -> float:
    return float(np.log(close).diff().tail(n).std() * math.sqrt(252))


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)


# --------------------------------------------------------------------------- #
# Probability helpers
# --------------------------------------------------------------------------- #
def prob_above(S: float, K: float, sigma: float, T: float, mu: float = 0.0) -> float:
    """P(price > K at time T) under lognormal with annual vol sigma, T in years."""
    if sigma <= 0 or T <= 0:
        return float(S > K)
    d = (math.log(S / K) + (mu - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d))


def prob_touch(S: float, K: float, sigma: float, T: float) -> float:
    """P(price touches K at any time before T) - reflection principle approx."""
    p_end = prob_above(S, K, sigma, T) if K > S else 1 - prob_above(S, K, sigma, T)
    return min(1.0, 2 * p_end)


def prob_target_before_stop(S: float, target: float, stop: float) -> float:
    """Driftless random walk in log space: P(hit target before stop)."""
    return (math.log(S) - math.log(stop)) / (math.log(target) - math.log(stop))


def bs_price(S, K, T, r, sigma, kind="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if kind == "call" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, kind="call") -> dict:
    if T <= 0 or sigma <= 0:
        return {"delta": float(S > K), "theta": 0.0, "vega": 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf = norm.pdf(d1)
    if kind == "call":
        delta = norm.cdf(d1)
        theta = -S * pdf * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -S * pdf * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    return {"delta": delta, "theta": theta / 365, "vega": S * pdf * math.sqrt(T) / 100}


# --------------------------------------------------------------------------- #
# Swing setups
# --------------------------------------------------------------------------- #
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["High20"] = df["High"].rolling(20).max().shift()
    return df


def setup_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean series per setup. Add your own here - they flow through everything."""
    uptrend = (df["Close"] > df["SMA50"]) & (df["SMA50"] > df["SMA50"].shift(10))
    return {
        "Trend pullback": uptrend
        & ((df["Low"] - df["EMA20"]).abs() <= 0.5 * df["ATR"])
        & df["RSI"].between(40, 58),
        "Breakout": (df["Close"] > df["High20"])
        & (df["Volume"] > 1.5 * df["VolAvg20"])
        & (df["Close"] > df["SMA200"]),
        "Oversold bounce": (df["RSI"] < 32) & (df["Close"] > df["SMA200"]),
    }


def backtest_setup(df: pd.DataFrame, signal: pd.Series, stop_atr: float,
                   target_atr: float, max_days: int) -> dict:
    """Historical frequency: of past signals, how often target hit before stop?"""
    wins = losses = timeouts = 0
    rets = []
    hi, lo, cl, a = (df[c].to_numpy() for c in ("High", "Low", "Close", "ATR"))
    idx = np.flatnonzero(signal.fillna(False).to_numpy())
    last_exit = -1
    for i in idx:
        if i <= last_exit or i + 1 >= len(df):
            continue  # don't count overlapping trades
        entry, stop, tgt = cl[i], cl[i] - stop_atr * a[i], cl[i] + target_atr * a[i]
        outcome, j = None, i
        for j in range(i + 1, min(i + 1 + max_days, len(df))):
            if lo[j] <= stop:      # conservative: stop checked first on same bar
                outcome = "loss"
                rets.append(stop / entry - 1)
                break
            if hi[j] >= tgt:
                outcome = "win"
                rets.append(tgt / entry - 1)
                break
        if outcome is None:
            if j - i < max_days:   # ran out of history; trade still open
                continue
            timeouts += 1
            rets.append(cl[j] / entry - 1)
        elif outcome == "win":
            wins += 1
        else:
            losses += 1
        last_exit = j
    n = wins + losses + timeouts
    return {"n": n, "win_rate": wins / n if n else np.nan,
            "avg_ret": float(np.mean(rets)) if rets else np.nan}


def swing_scan(ticker: str, df: pd.DataFrame, stop_atr=1.5, target_atr=3.0,
               max_days=15) -> dict:
    df = add_indicators(df)
    last = df.iloc[-1]
    sigs = setup_signals(df)
    active = [name for name, s in sigs.items() if bool(s.iloc[-1])]
    S, a = float(last["Close"]), float(last["ATR"])
    stop, target = S - stop_atr * a, S + target_atr * a
    sigma = realized_vol(df["Close"])
    T = max_days / 252
    # Historical edge for each setup (whether active today or not).
    stats = {name: backtest_setup(df, s, stop_atr, target_atr, max_days) for name, s in sigs.items()}
    best = max(active, key=lambda n: stats[n]["win_rate"] if stats[n]["n"] else -1) if active else None
    trend = ("Up" if S > last["SMA50"] > last["SMA200"]
             else "Down" if S < last["SMA50"] < last["SMA200"] else "Mixed")
    return {
        "Ticker": ticker, "Price": S, "Trend": trend, "RSI": float(last["RSI"]),
        "ATR %": a / S * 100, "Setups today": ", ".join(active) or "—",
        "Entry": S, "Stop": stop, "Target": target,
        "P(target before stop)": prob_target_before_stop(S, target, stop),
        "P(touch target)": prob_touch(S, target, sigma, T),
        "Hist win rate": float(stats[best]["win_rate"]) if best else np.nan,
        "Hist signals": stats[best]["n"] if best else 0,
        "_stats": stats,
    }


# --------------------------------------------------------------------------- #
# LEAPS
# --------------------------------------------------------------------------- #
def leaps_table(chain: pd.DataFrame, spot: float, r: float = 0.04,
                kind: str = "call", hist_vol: float | None = None) -> pd.DataFrame:
    today = dt.date.today()
    c = chain[(chain["type"] == kind) & (chain["mid"] > 0) & (chain["iv"] > 0.01)].copy()
    if c.empty:
        return c
    out = []
    for _, row in c.iterrows():
        T = max((pd.Timestamp(row["expiry"]).date() - today).days, 1) / 365
        K, mid, iv = float(row["strike"]), float(row["mid"]), float(row["iv"])
        g = bs_greeks(spot, K, T, r, iv, kind)
        be = K + mid if kind == "call" else K - mid
        intrinsic = max(spot - K, 0) if kind == "call" else max(K - spot, 0)
        sig = hist_vol or iv
        p_itm = prob_above(spot, K, sig, T) if kind == "call" else 1 - prob_above(spot, K, sig, T)
        p_profit = prob_above(spot, be, sig, T) if kind == "call" else 1 - prob_above(spot, be, sig, T)
        spread_pct = (row["ask"] - row["bid"]) / mid * 100 if mid else np.nan
        out.append({
            "Expiry": row["expiry"], "DTE": int(T * 365), "Strike": K, "Mid": mid,
            "IV %": iv * 100, "Delta": g["delta"], "Theta/day": g["theta"],
            "Breakeven": be, "BE move %": (be / spot - 1) * 100,
            "Extrinsic %": (mid - intrinsic) / mid * 100,
            "P(ITM)": p_itm, "P(profit @ exp)": p_profit,
            "Leverage": abs(g["delta"]) * spot / mid,
            "Spread %": spread_pct, "OI": int(row.get("oi") or 0),
        })
    df = pd.DataFrame(out)
    # LEAPS "fit": deep-ish ITM (delta .70-.85), low extrinsic, tight spread, liquid.
    delta_fit = 1 - (df["Delta"].abs() - 0.78).abs() / 0.3
    df["Fit score"] = (40 * delta_fit.clip(0, 1)
                       + 30 * (1 - df["Extrinsic %"].clip(0, 40) / 40)
                       + 15 * (1 - df["Spread %"].fillna(20).clip(0, 10) / 10)
                       + 15 * (np.log1p(df["OI"]) / np.log1p(5000)).clip(0, 1)).round(0)
    df = df[["Fit score"] + [c for c in df.columns if c != "Fit score"]]
    return df.sort_values("Fit score", ascending=False)


# --------------------------------------------------------------------------- #
# Earnings
# --------------------------------------------------------------------------- #
def earnings_reactions(daily: pd.DataFrame, report_times: list[pd.Timestamp]) -> pd.DataFrame:
    """Close-to-close move on the first session that could react to each report."""
    rows = []
    idx = daily.index
    for ts in report_times:
        d = pd.Timestamp(ts.date())
        after_close = ts.hour >= 12
        pos = idx.searchsorted(d + pd.Timedelta(days=1 if after_close else 0))
        if pos <= 0 or pos >= len(idx):
            continue
        prev_c, o, c = daily["Close"].iloc[pos - 1], daily["Open"].iloc[pos], daily["Close"].iloc[pos]
        rows.append({"Report": ts.strftime("%Y-%m-%d") + (" AMC" if after_close else " BMO"),
                     "Gap %": (o / prev_c - 1) * 100, "Day move %": (c / prev_c - 1) * 100})
    return pd.DataFrame(rows)


def implied_move(chain: pd.DataFrame, spot: float) -> float:
    """ATM straddle mid / spot, in %."""
    if chain.empty:
        return np.nan
    k = chain.loc[(chain["strike"] - spot).abs().idxmin(), "strike"]
    atm = chain[chain["strike"] == k]
    return float(atm["mid"].sum() / spot * 100) if len(atm) == 2 else np.nan


# --------------------------------------------------------------------------- #
# Day trading
# --------------------------------------------------------------------------- #
def day_levels(intra: pd.DataFrame, daily: pd.DataFrame) -> dict:
    dates = sorted(set(intra.index.date))
    today = dates[-1]
    tod = intra.index.time
    reg = (tod >= dt.time(9, 30)) & (tod < dt.time(16, 0))
    prev = intra[reg & (intra.index.date == dates[-2])] if len(dates) > 1 else intra.iloc[:0]
    t = intra[intra.index.date == today]
    pm = t[t.index.time < dt.time(9, 30)]
    rth = t[(t.index.time >= dt.time(9, 30)) & (t.index.time < dt.time(16, 0))]
    orb = rth[rth.index.time < dt.time(10, 0)]  # 30-minute opening range
    lv: dict[str, float] = {}
    if len(prev):
        H, L, C = prev["High"].max(), prev["Low"].min(), prev["Close"].iloc[-1]
        P = (H + L + C) / 3
        lv.update({"PDH": H, "PDL": L, "PDC": C, "Pivot": P,
                   "R1": 2 * P - L, "S1": 2 * P - H, "R2": P + (H - L), "S2": P - (H - L)})
    if len(pm):
        lv.update({"PMH": pm["High"].max(), "PML": pm["Low"].min()})
    if len(orb):
        lv.update({"ORH": orb["High"].max(), "ORL": orb["Low"].min()})
    a = float(atr(daily).iloc[-1])
    stats = {"ATR": a}
    if len(rth):
        stats["ATR used %"] = (rth["High"].max() - rth["Low"].min()) / a * 100
        # RVOL: today's RTH volume so far vs same time-of-day on prior days.
        cutoff = rth.index[-1].time()
        past = []
        for d in dates[:-1]:
            p = intra[(intra.index.date == d) & reg & (intra.index.time <= cutoff)]
            if len(p):
                past.append(p["Volume"].sum())
        stats["RVOL"] = rth["Volume"].sum() / np.mean(past) if past and np.mean(past) else np.nan
    if len(pm) and "PDC" in lv:
        stats["Gap %"] = (pm["Close"].iloc[-1] / lv["PDC"] - 1) * 100
    return {"levels": lv, "stats": stats, "today": t, "rth": rth}
