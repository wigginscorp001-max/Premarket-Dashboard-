"""Day-by-day options backtest engine (long calls / long puts).

Timing (no look-ahead):
  * Signals use day t's close.  Entries/exits that come from a signal fill on day t+1.
  * Fills use that day's ORATS snapshot: BUY at the ASK, SELL at the BID, plus commission.
  * Profit target / stop loss / time stop are checked once per day on the snapshot (bid).
    Daily data cannot see intraday touches, so stops fill at the day's bid – usually worse
    than the stop level (conservative).
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from . import signals as sig


@dataclass
class Params:
    tickers: list[str]
    start: pd.Timestamp
    end: pd.Timestamp
    strategy: str = "Swing (45–60 DTE)"
    direction: str = "Calls & Puts"          # "Calls only" | "Puts only" | "Calls & Puts"
    dte_min: int = 45
    dte_max: int = 60
    delta_min: float = 0.60
    delta_max: float = 0.70
    max_spread_pct: float = 10.0             # (ask-bid)/mid, %
    min_oi: int = 100
    profit_target: float | None = 50.0       # % gain on premium
    stop_loss: float | None = 40.0           # % loss on premium
    time_stop_dte: int | None = 21           # close when DTE <= this
    max_hold_days: int | None = None
    reversal_exit: bool = True
    reversal_rule: str = sig.REVERSAL_RULES[0]
    earnings_mode: str = "Skip"              # "Skip" | "Hold through" | "Exit before"
    commission: float = 0.65                 # $ per contract per side
    slippage: float = 0.0                    # $ per share added to buys / taken from sells
    capital: float = 100_000.0
    alloc_pct: float = 5.0                   # % of equity per trade
    max_positions: int = 5
    rules: sig.EntryRules = field(default_factory=sig.EntryRules)
    market_ticker: str = "SPY"
    iv_regime_tags: bool = True              # tag each trade High/Low IV (needs IV rank history)

    @property
    def target_delta(self) -> float:
        return (self.delta_min + self.delta_max) / 2

    def chain_window(self) -> tuple[int, int]:
        lo = min(self.dte_min, self.time_stop_dte or 0)
        return max(0, lo - 3), self.dte_max + 5


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.DataFrame
    params: Params
    notes: list[str]
    skipped: dict


def select_contract(ch: pd.DataFrame, typ: str, p: Params) -> pd.Series | None:
    if ch is None or ch.empty:
        return None
    c = ch[(ch["type"] == typ) & (ch["dte"] >= p.dte_min) & (ch["dte"] <= p.dte_max)]
    c = c[(c["bid"] > 0) & (c["ask"] > 0) & (c["ask"] >= c["bid"])]
    ad = c["delta"].abs()
    c = c[(ad >= p.delta_min) & (ad <= p.delta_max)]
    if c.empty:
        return None
    mid = (c["bid"] + c["ask"]) / 2
    c = c[((c["ask"] - c["bid"]) / mid * 100 <= p.max_spread_pct)]
    if p.min_oi:
        c = c[c["oi"].fillna(0) >= p.min_oi]
    if c.empty:
        return None
    mid_dte = (p.dte_min + p.dte_max) / 2
    score = (c["delta"].abs() - p.target_delta).abs() + 0.002 * (c["dte"] - mid_dte).abs()
    return c.loc[score.idxmin()]


def _quote(ch: pd.DataFrame, pos: dict) -> pd.Series | None:
    if ch is None or ch.empty:
        return None
    m = ((ch["type"].values == pos["type"]) & (ch["expiry"].values == np.datetime64(pos["expiry"], "ns"))
         & np.isclose(ch["strike"].values.astype(float), pos["strike"]))
    idx = np.flatnonzero(m)
    return None if len(idx) == 0 else ch.iloc[idx[0]]


def run(p: Params, source, progress=None) -> Result:
    notes, skipped = [], {"no contract": 0, "earnings": 0, "max positions": 0, "too expensive": 0, "no cash": 0}
    say = progress or (lambda frac, msg: None)
    lo_w, hi_w = p.chain_window()

    # ---- 1. stock data + signals (all known at each day's close) -----------------------
    feats, earn, sig_c, sig_p, rev_c, rev_p = {}, {}, {}, {}, {}, {}
    for i, t in enumerate(p.tickers):
        say(0.02 + 0.08 * i / max(len(p.tickers), 1), f"Loading prices for {t}…")
        px = source.prices(t)
        ivr = None
        if p.rules.ivr_filter or p.iv_regime_tags:
            try:
                ivr = source.iv_rank(t, p.start, p.end,
                                     lambda f, m, i=i: say(0.02 + 0.08 * (i + f) / max(len(p.tickers), 1), m))
            except Exception as e:
                notes.append(f"{t}: IV rank unavailable ({e}).")
        orb = None
        if p.rules.orb_filter:
            orb = source.opening_range(t, p.start - pd.Timedelta(days=5), p.end)
            if orb is None or orb.empty:
                notes.append(f"{t}: no opening-range data (needs Alpha Vantage, or POLYGON_API_KEY with ORATS + FMP) – no ORB trades possible.")
        f = ind.features(px, ivr, orb, p.rules.breakout_len)
        feats[t] = f
        sig_c[t], sig_p[t] = sig.entry_signals(f, p.rules)
        rev_c[t], rev_p[t] = sig.reversal_signals(f, p.reversal_rule)
        try:
            earn[t] = pd.DatetimeIndex(source.earnings(t))
        except Exception as e:
            earn[t] = pd.DatetimeIndex([])
            notes.append(f"{t}: earnings dates unavailable ({e}).")

    try:
        mkt = ind.features(source.prices(p.market_ticker))
        bull = (mkt["close"] > ind.sma(mkt["close"], 200))
    except Exception:
        bull = pd.Series(dtype=bool)
        notes.append(f"Market regime unavailable ({p.market_ticker} prices failed).")

    cal = sorted(set().union(*[set(f.index) for f in feats.values()]))
    cal = pd.DatetimeIndex([d for d in cal if p.start <= d <= p.end])
    if len(cal) < 2:
        raise ValueError("Not enough trading days in the chosen date range.")

    want_c = p.direction in ("Calls only", "Calls & Puts")
    want_p = p.direction in ("Puts only", "Calls & Puts")

    # ---- 2. prefetch chains on every possible entry day (parallel, cached) ------------
    entry_days = []
    for t in p.tickers:
        s = pd.Series(False, index=feats[t].index)
        if want_c: s |= sig_c[t]
        if want_p: s |= sig_p[t]
        nxt = feats[t].index.to_series().shift(-1)
        for d in nxt[s & (s.index >= cal[0] - pd.Timedelta(days=7))].dropna():
            if cal[0] <= d <= cal[-1]:
                entry_days.append((t, d))
    chain_cache: dict = {}

    def get_chain(t, d):
        k = (t, d)
        if k not in chain_cache:
            try:
                chain_cache[k] = source.chain(t, d, lo_w, hi_w)
            except Exception as e:
                chain_cache[k] = None
                if len(notes) < 40:
                    notes.append(f"{t} {d.date()}: chain failed ({e}).")
        return chain_cache[k]

    def prefetch(keys, base, span, label):
        keys = [k for k in keys if k not in chain_cache]
        if not keys:
            return
        with ThreadPoolExecutor(max_workers=8) as ex:
            for j, _ in enumerate(ex.map(lambda k: get_chain(*k), keys)):
                if j % 10 == 0:
                    say(base + span * j / len(keys), f"{label} {j}/{len(keys)}")

    prefetch(entry_days, 0.10, 0.30, "Downloading option chains (entry days)")

    # ---- 3. day loop ---------------------------------------------------------------------
    cash, open_pos, trades, curve = p.capital, [], [], []
    for i in range(1, len(cal)):
        d, prev = cal[i], cal[i - 1]
        if i % 20 == 0:
            say(0.40 + 0.58 * i / len(cal), f"Simulating {d.strftime('%d/%b/%Y')}")

        # 3a. manage open positions
        still = []
        for pos in open_pos:
            t = pos["ticker"]
            q = _quote(get_chain(t, d), pos)
            reason = None
            if q is None:
                if d >= pos["expiry"]:
                    col = "close_raw" if "close_raw" in feats[t].columns else "close"
                    spot = feats[t][col].get(d, np.nan)
                    intrinsic = max(0.0, (spot - pos["strike"]) if pos["type"] == "C" else (pos["strike"] - spot))
                    q = pd.Series({"bid": intrinsic, "ask": intrinsic, "dte": 0})
                    reason = "Expired"
                else:
                    still.append(pos)
                    continue
            bid, mid = float(q["bid"]), (float(q["bid"]) + float(q["ask"])) / 2
            pos["mark"] = mid
            chg = (bid - pos["entry_px"]) / pos["entry_px"] * 100
            held = (d - pos["entry_date"]).days
            f_prev = feats[t]
            if reason is None:
                if p.stop_loss and chg <= -p.stop_loss:
                    reason = "Stop loss"
                elif p.profit_target and chg >= p.profit_target:
                    reason = "Profit target"
                elif p.time_stop_dte and int(q["dte"]) <= p.time_stop_dte:
                    reason = "Time stop"
                elif p.earnings_mode == "Exit before" and _earnings_next_day(earn[t], d, cal, i):
                    reason = "Earnings exit"
                elif p.reversal_exit and prev in f_prev.index and bool(
                        (rev_c if pos["type"] == "C" else rev_p)[t].get(prev, False)) and prev > pos["entry_date"]:
                    reason = "Signal reversal"
                elif p.max_hold_days and held >= p.max_hold_days:
                    reason = "Max hold"
                elif i == len(cal) - 1:
                    reason = "End of test"
            if reason:
                fill = max(bid - p.slippage, 0.0)
                fee = p.commission * pos["contracts"]
                proceeds = fill * 100 * pos["contracts"] - fee
                cash += proceeds
                pnl = proceeds - pos["cost"]
                trades.append({**{k: v for k, v in pos.items() if k != "mark"},
                               "exit_date": d, "exit_px": fill, "exit_reason": reason,
                               "proceeds": proceeds, "pnl": pnl, "pnl_pct": pnl / pos["cost"] * 100,
                               "days_held": held, "commission_total": pos["entry_fee"] + fee})
            else:
                still.append(pos)
        open_pos = still

        # 3b. new entries from yesterday's close signals
        if i < len(cal) - 1:
            for t in p.tickers:
                f = feats[t]
                if prev not in f.index:
                    continue
                for typ, s, ok in (("C", sig_c[t], want_c), ("P", sig_p[t], want_p)):
                    if not ok or not bool(s.get(prev, False)):
                        continue
                    if any(o["ticker"] == t for o in open_pos):
                        continue
                    if len(open_pos) >= p.max_positions:
                        skipped["max positions"] += 1
                        continue
                    ch = get_chain(t, d)
                    k = select_contract(ch, typ, p)
                    if k is None:
                        skipped["no contract"] += 1
                        continue
                    if p.earnings_mode == "Skip" and _earnings_in_window(earn[t], d, k["expiry"], p):
                        skipped["earnings"] += 1
                        continue
                    if p.earnings_mode == "Exit before" and bool(
                            ((earn[t] >= d) & (earn[t] <= d + pd.Timedelta(days=3))).any()):
                        skipped["earnings"] += 1
                        continue
                    price = float(k["ask"]) + p.slippage
                    equity_now = cash + sum(o["mark"] * 100 * o["contracts"] for o in open_pos)
                    n = math.floor(equity_now * p.alloc_pct / 100 / (price * 100 + p.commission))
                    if n < 1:
                        skipped["too expensive"] += 1
                        continue
                    fee = p.commission * n
                    cost = price * 100 * n + fee
                    if cost > cash:
                        skipped["no cash"] += 1
                        continue
                    cash -= cost
                    ivr = f["ivr"].get(prev, np.nan)
                    pos = {"ticker": t, "type": typ, "entry_date": d, "signal_date": prev,
                           "expiry": k["expiry"], "strike": float(k["strike"]), "dte_entry": int(k["dte"]),
                           "delta_entry": float(k["delta"]), "spot_entry": float(k["spot"]) if pd.notna(k["spot"]) else np.nan,
                           "entry_px": price, "contracts": n, "cost": cost, "entry_fee": fee,
                           "mark": (float(k["bid"]) + float(k["ask"])) / 2, "ivr_entry": ivr,
                           "market": ("Bull" if bool(bull.get(prev, False)) else "Bear") if len(bull) else "n/a",
                           "iv_regime": "n/a" if pd.isna(ivr) else ("High IV" if ivr >= 50 else "Low IV")}
                    open_pos.append(pos)
                    # prefetch the days this contract can be held
                    last = min(k["expiry"], cal[-1])
                    hold_days = [x for x in cal[i + 1:] if x <= last][:400]
                    prefetch([(t, x) for x in hold_days], 0.40 + 0.58 * i / len(cal), 0.0, "Downloading holding days")
                    break  # one position per ticker

        mtm = sum(o["mark"] * 100 * o["contracts"] for o in open_pos)
        curve.append({"date": d, "cash": cash, "positions": mtm, "equity": cash + mtm, "open": len(open_pos)})

    say(1.0, "Done")
    tr = pd.DataFrame(trades)
    if not tr.empty:
        tr["year"] = tr["entry_date"].dt.year
        tr["win"] = tr["pnl"] > 0
        tr = tr.sort_values("entry_date").reset_index(drop=True)
    eq = pd.DataFrame(curve).set_index("date")
    return Result(tr, eq, p, notes, skipped)


def _earnings_in_window(dates: pd.DatetimeIndex, entry: pd.Timestamp, expiry: pd.Timestamp, p: Params) -> bool:
    if len(dates) == 0:
        return False
    end = expiry - pd.Timedelta(days=p.time_stop_dte or 0)
    if p.max_hold_days:
        end = min(end, entry + pd.Timedelta(days=p.max_hold_days))
    return bool(((dates >= entry) & (dates <= end)).any())


def _earnings_next_day(dates: pd.DatetimeIndex, d: pd.Timestamp, cal: pd.DatetimeIndex, i: int) -> bool:
    """True if earnings fall on the next trading day (exit today, before the report)."""
    if len(dates) == 0:
        return False
    nxt = cal[i + 1] if i + 1 < len(cal) else d + pd.Timedelta(days=1)
    return bool(((dates > d) & (dates <= nxt)).any())
