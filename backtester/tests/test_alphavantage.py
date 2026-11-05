"""Alpha Vantage source tested with fake responses in Alpha Vantage's exact JSON format."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from bt import alphavantage as av
from bt import demo, engine, store
from bt import signals as sig
from bt.sources import AlphaVantageSource


def fake_get(function, **p):
    t = p.get("symbol")
    if function == "TIME_SERIES_DAILY_ADJUSTED":
        px = demo.daily_prices(t)
        ts = {d.strftime("%Y-%m-%d"): {"1. open": f"{r.open:.4f}", "2. high": f"{r.high:.4f}",
                                       "3. low": f"{r.low:.4f}", "4. close": f"{r.close:.4f}",
                                       "5. adjusted close": f"{r.close:.4f}", "6. volume": f"{int(r.volume)}",
                                       "7. dividend amount": "0.0000", "8. split coefficient": "1.0"}
              for d, r in px.iterrows()}
        return {"Meta Data": {}, "Time Series (Daily)": ts}
    if function == "EARNINGS":
        return {"symbol": t, "quarterlyEarnings": [{"reportedDate": d.strftime("%Y-%m-%d")}
                                                    for d in demo.earnings_dates(t)]}
    if function == "HISTORICAL_OPTIONS":
        d = pd.Timestamp(p["date"])
        ch = demo.chain(t, d, 0, 800)
        rows = [{"contractID": "X", "symbol": t, "expiration": r.expiry.strftime("%Y-%m-%d"),
                 "strike": f"{r.strike:.2f}", "type": "call" if r.type == "C" else "put",
                 "last": "0", "mark": f"{(r.bid + r.ask) / 2:.2f}", "bid": f"{r.bid:.2f}", "bid_size": "4",
                 "ask": f"{r.ask:.2f}", "ask_size": "5", "volume": "10", "open_interest": f"{int(r.oi)}",
                 "date": p["date"], "implied_volatility": f"{r.iv:.5f}", "delta": f"{r.delta:.5f}",
                 "gamma": "0", "theta": "0", "vega": "0", "rho": "0"} for r in ch.itertuples()]
        return {"endpoint": "Historical Options", "message": "success", "data": rows}
    if function == "TIME_SERIES_INTRADAY":
        m = pd.Period(p["month"])
        o = demo.opening_range(t, None, None)
        o = o[(o.index >= m.start_time) & (o.index <= m.end_time)]
        ts = {}
        for d, r in o.iterrows():
            ts[(d + pd.Timedelta(hours=9, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")] = {
                "1. open": "0", "2. high": f"{r.orh}", "3. low": f"{r.orl}", "4. close": "0", "5. volume": "0"}
            ts[(d + pd.Timedelta(hours=10)).strftime("%Y-%m-%d %H:%M:%S")] = {
                "1. open": "0", "2. high": "99999", "3. low": "0", "4. close": "0", "5. volume": "0"}
        return {"Time Series (30min)": ts}
    raise AssertionError(function)


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    store._mem.clear()
    monkeypatch.setattr(av, "_get", fake_get)
    monkeypatch.setattr(store, "put", lambda k, df: store._remember(k, (df, pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=None))))
    monkeypatch.setattr(store, "get", lambda k: store._mem.get(k))
    yield


def test_normalise_matches_screenshot_format():
    rows = [{"contractID": "IBM171117C00075000", "symbol": "IBM", "expiration": "2017-11-17", "strike": "75.00",
             "type": "call", "last": "38.35", "mark": "72.28", "bid": "70.45", "bid_size": "4", "ask": "74.10",
             "ask_size": "5", "volume": "0", "open_interest": "0", "date": "2017-11-15",
             "implied_volatility": "4.20990", "delta": "0.98977", "gamma": "0.00059", "theta": "-0.31401",
             "vega": "0.00296", "rho": "0.00402"},
            {"contractID": "IBM171117P00075000", "symbol": "IBM", "expiration": "2017-11-17", "strike": "75.00",
             "type": "put", "bid": "0.00", "ask": "0.03", "open_interest": "0", "implied_volatility": "3.05871",
             "delta": "-0.00101"}]
    n = av.normalise(rows, pd.Timestamp("2017-11-15"))
    assert list(n["type"]) == ["C", "P"]
    c = n.iloc[0]
    assert c["dte"] == 2 and c["bid"] == 70.45 and c["ask"] == 74.10 and abs(c["delta"] - 0.98977) < 1e-9


def test_split_adjustment():
    raw = pd.DataFrame({"date": ["2020-01-02", "2020-01-03", "2020-01-06"], "open": [200, 202, 51],
                        "high": [200, 202, 51], "low": [200, 202, 51], "close": [200, 202, 51],
                        "volume": [100, 100, 400], "split": [1.0, 1.0, 4.0]})
    av_daily = av._daily_raw
    try:
        av._daily_raw = lambda t: raw
        px = av.daily_prices("X")
    finally:
        av._daily_raw = av_daily
    assert list(px["close"].round(2)) == [50.0, 50.5, 51.0]       # pre-split prices divided by 4
    assert list(px["close_raw"]) == [200, 202, 51]
    assert list(px["volume"]) == [400, 400, 400]


def test_opening_range_uses_first_bar_only():
    o = av.opening_range("AAA", pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-28"))
    ref = demo.opening_range("AAA", None, None).loc[o.index]
    assert len(o) > 30 and np.allclose(o["orh"], ref["orh"])        # 10:00 bar (99999) ignored


def test_full_backtest_through_alpha_vantage():
    p = engine.Params(tickers=["AAA"], start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                      market_ticker="AAA", rules=sig.EntryRules(trend_pullback=True, ivr_filter=False))
    r = engine.run(p, AlphaVantageSource())
    tr = r.trades
    assert len(tr) > 0, r.notes
    assert tr["dte_entry"].between(45, 60).all()
    assert tr["delta_entry"].abs().between(0.60, 0.70).all()
    assert (tr["entry_date"] > tr["signal_date"]).all()
    assert tr["iv_regime"].isin(["High IV", "Low IV", "n/a"]).all()
    assert (tr["iv_regime"] != "n/a").any()                       # IV rank was built from weekly chains
