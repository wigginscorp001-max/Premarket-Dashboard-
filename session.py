"""Pre-market setup -> opening range -> intraday -> afternoon, for one day or many.

Works on 5-minute bars with columns Open/High/Low/Close/Volume and an ET index
(the format data.intraday() returns). The behavior rules are the ones from
pm_vs_intraday_behavior.py in the TradingJournal folder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAN = float("nan")
PM, OPEN, ORB_END, NOON, AFTERNOON, CLOSE = 240, 570, 600, 720, 780, 960  # minutes after midnight


def _minutes(df: pd.DataFrame) -> np.ndarray:
    return df.index.hour * 60 + df.index.minute


def classify_behavior(pm_bias: str, orb_high: float, orb_low: float, intraday: pd.DataFrame) -> str:
    close_price = intraday["Close"].iloc[-1]
    hi, lo = intraday["High"].max(), intraday["Low"].min()
    if close_price > orb_high and pm_bias == "BULLISH":
        return "Breakout Continuation"
    if close_price < orb_low and pm_bias == "BEARISH":
        return "Breakdown Continuation"
    if hi > orb_high and lo < orb_low:
        return "Choppy / Range Expansion"
    if orb_low <= close_price <= orb_high:
        return "Inside Opening Range"
    if close_price > orb_high:
        return "Bullish Reversal"
    return "Bearish Reversal"


def analyze_day(day: pd.DataFrame, prev_close: float = NAN) -> dict:
    """Everything for one session. Fields fill in as the day progresses."""
    m = _minutes(day)
    pm = day[(m >= PM) & (m < OPEN)]
    orb = day[(m >= OPEN) & (m < ORB_END)]
    rth = day[(m >= OPEN) & (m < CLOSE)]
    intraday = day[(m >= ORB_END) & (m < CLOSE)]
    morning = day[(m >= OPEN) & (m < NOON)]
    afternoon = day[(m >= AFTERNOON) & (m < CLOSE)]
    row: dict = {"Prev close": prev_close}
    if pm.empty:
        row["Behavior"] = "no pre-market trades"
        return row
    traded = pm[pm["Volume"] > 0]
    traded = traded if len(traded) else pm
    vol = traded["Volume"].sum()
    tp = (traded["High"] + traded["Low"] + traded["Close"]) / 3
    pm_vwap = (tp * traded["Volume"]).sum() / vol if vol else traded["Close"].mean()
    pm_last = traded["Close"].iloc[-1]
    bias = "BULLISH" if pm_last > pm_vwap else "BEARISH"
    row.update({
        "PM last": pm_last,
        "Gap %": (pm_last / prev_close - 1) * 100 if prev_close == prev_close else NAN,
        "PM high": traded["High"].max(), "PM low": traded["Low"].min(),
        "PM VWAP": pm_vwap, "PM bias": bias, "PM volume": int(vol),
        "Last PM print": traded.index[-1].strftime("%H:%M"),
    })
    if orb.empty:
        row["Behavior"] = "Pre-market live – behavior shows after 10:00"
        return row
    row.update({"ORB high": orb["High"].max(), "ORB low": orb["Low"].min(),
                "Open": orb["Open"].iloc[0]})
    if intraday.empty:
        row["Behavior"] = "opening range forming"
        return row
    row["Price"] = rth["Close"].iloc[-1]
    row["Behavior"] = classify_behavior(bias, row["ORB high"], row["ORB low"], intraday)
    row["Took PM high"] = bool(rth["High"].max() > row["PM high"])
    row["Took PM low"] = bool(rth["Low"].min() < row["PM low"])
    # Opening-range breakout: first 5-min bar after 10:00 to trade through each side.
    orb_rng = row["ORB high"] - row["ORB low"]
    up_hits = intraday[intraday["High"] > row["ORB high"]]
    dn_hits = intraday[intraday["Low"] < row["ORB low"]]
    up_t = up_hits.index[0] if len(up_hits) else None
    dn_t = dn_hits.index[0] if len(dn_hits) else None
    if up_t is not None and (dn_t is None or up_t <= dn_t):
        row["ORB break"] = f"Up {up_t:%H:%M}" + (f", then down {dn_t:%H:%M}" if dn_t is not None else "")
    elif dn_t is not None:
        row["ORB break"] = f"Down {dn_t:%H:%M}" + (f", then up {up_t:%H:%M}" if up_t is not None else "")
    else:
        row["ORB break"] = "No break"
    row["Broke ORB high"], row["Broke ORB low"] = up_t is not None, dn_t is not None
    row["ORB range %"] = orb_rng / row["Open"] * 100
    if orb_rng > 0:
        # how far price ran past the range, in multiples of the range (1.0 = a full range)
        row["Ext above ORB (x)"] = max(intraday["High"].max() - row["ORB high"], 0) / orb_rng
        row["Ext below ORB (x)"] = max(row["ORB low"] - intraday["Low"].min(), 0) / orb_rng
    if len(morning):
        row["Morning %"] = (morning["Close"].iloc[-1] / row["Open"] - 1) * 100
    if len(afternoon):
        a_open = afternoon["Open"].iloc[0]
        row["Afternoon %"] = (afternoon["Close"].iloc[-1] / a_open - 1) * 100
        row["Afternoon high"], row["Afternoon low"] = afternoon["High"].max(), afternoon["Low"].min()
        followed = row["Afternoon %"] > 0 if bias == "BULLISH" else row["Afternoon %"] < 0
        row["Afternoon followed PM?"] = "Yes" if followed else "No"
        row["Close vs PM VWAP"] = "Above" if row["Price"] > pm_vwap else "Below"
    finished = m[-1] >= CLOSE - 5
    row["Day complete"] = bool(finished)
    if not finished:
        row["Behavior"] += " (so far)"
    return row


def split_days(df: pd.DataFrame) -> list[tuple]:
    """[(date, that day's bars, prior regular-session close), ...] oldest first."""
    m = _minutes(df)
    regular = df[(m >= OPEN) & (m < CLOSE)]
    reg_dates = regular.index.date
    out = []
    for day in sorted(set(df.index.date)):
        prior = regular[reg_dates < day]
        out.append((day, df[df.index.date == day], prior["Close"].iloc[-1] if len(prior) else NAN))
    return out


def today(ticker: str, bars: pd.DataFrame) -> dict:
    if bars is None or bars.empty:
        return {"Ticker": ticker, "Behavior": "No data – check the symbol"}
    day, g, prev_close = split_days(bars)[-1]
    return {"Ticker": ticker, "Session date": day, **analyze_day(g, prev_close)}


def history(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per completed past session."""
    rows = []
    for day, g, pc in split_days(bars):
        r = analyze_day(g, pc)
        if r.get("Day complete"):
            rows.append({"Date": day, **r})
    return pd.DataFrame(rows)


def odds(hist: pd.DataFrame, bias: str | None = None) -> dict:
    """How past sessions with the same pre-market bias played out."""
    if hist.empty or "PM bias" not in hist:
        return {}
    h = hist if bias is None else hist[hist["PM bias"] == bias]
    if h.empty:
        return {}
    cont = {"BULLISH": "Breakout Continuation", "BEARISH": "Breakdown Continuation"}
    is_cont = h.apply(lambda r: r["Behavior"].startswith(cont[r["PM bias"]]), axis=1)
    out = {
        "Sessions": len(h),
        "Continuation %": is_cont.mean() * 100,
        "Chop %": (h["Behavior"] == "Choppy / Range Expansion").mean() * 100,
        "Broke ORB high %": h["Broke ORB high"].mean() * 100,
        "Broke ORB low %": h["Broke ORB low"].mean() * 100,
        "Took PM high %": h["Took PM high"].mean() * 100,
        "Took PM low %": h["Took PM low"].mean() * 100,
        "Most common": h["Behavior"].mode().iloc[0],
    }
    if "Afternoon followed PM?" in h:
        out["Afternoon follows PM %"] = (h["Afternoon followed PM?"] == "Yes").mean() * 100
    return out
