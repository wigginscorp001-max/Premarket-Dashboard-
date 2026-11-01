import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import os, tempfile
os.environ.pop("DATABASE_URL", None)

import pandas as pd
from bt import engine, metrics, signals as sig, orats
from bt.sources import DemoSource


def _run(**kw):
    p = engine.Params(tickers=["AAA", "BBB"], start=pd.Timestamp("2016-01-01"), end=pd.Timestamp("2019-12-31"),
                      market_ticker="SPY", **kw)
    return engine.run(p, DemoSource())


def test_swing_runs_and_respects_rules():
    r = _run()
    tr = r.trades
    assert len(tr) > 5
    assert (tr["dte_entry"].between(45, 60)).all()
    assert (tr["delta_entry"].abs().between(0.60, 0.70)).all()
    assert (tr["entry_date"] > tr["signal_date"]).all()          # no look-ahead
    assert (tr["exit_date"] >= tr["entry_date"]).all()
    # equity reconciles with realised P&L at the end (all closed by "End of test")
    final = r.equity["equity"].iloc[-1]
    assert abs(final - (r.params.capital + tr["pnl"].sum())) < 1.0
    s = metrics.summary(tr, r.equity, r.params.capital)
    assert 0 <= s["win_rate"] <= 100 and s["max_dd_pct"] <= 0


def test_leaps_and_breakdowns():
    r = _run(strategy="LEAPS", dte_min=365, dte_max=730, delta_min=0.70, delta_max=0.85,
             profit_target=100, stop_loss=35, time_stop_dte=120, direction="Calls only", earnings_mode="Hold through",
             rules=sig.EntryRules(trend_pullback=False, ema_trend=True))
    assert len(r.trades) > 0
    assert set(r.trades["type"]) == {"C"}
    for by in ["year", "ticker", "market", "iv_regime", "exit_reason"]:
        assert not metrics.breakdown(r.trades, by).empty


def test_orats_normalise():
    raw = pd.DataFrame([{"ticker": "X", "tradeDate": "2020-01-02", "expirDate": "2020-02-21", "dte": 50,
                         "strike": 100, "stockPrice": 101, "callBidPrice": 4, "callAskPrice": 4.2,
                         "putBidPrice": 3, "putAskPrice": 3.2, "delta": 0.55, "callOpenInterest": 10,
                         "putOpenInterest": 20, "smvVol": 0.3}])
    n = orats.normalise(raw)
    assert len(n) == 2
    put = n[n["type"] == "P"].iloc[0]
    assert abs(put["delta"] + 0.45) < 1e-9 and put["bid"] == 3
