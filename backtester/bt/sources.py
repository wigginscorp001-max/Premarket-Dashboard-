"""One interface over the paid feeds (FMP + ORATS + optional Polygon) or demo data."""
from __future__ import annotations

import pandas as pd

from . import alphavantage as av
from . import demo, fmp, orats, polygon


class LiveSource:
    name = "Live (ORATS + FMP)"
    synthetic = False

    def prices(self, t):            return fmp.daily_prices(t)
    def earnings(self, t):          return fmp.earnings_dates(t)
    def iv_rank(self, t, *a, **k):  return orats.iv_rank(t)
    def chain(self, t, d, lo, hi):  return orats.chain(t, d, lo, hi)
    def opening_range(self, t, s, e): return polygon.opening_range(t, s, e)


class DemoSource:
    name = "Demo (synthetic – not real data)"
    synthetic = True

    def prices(self, t):            return demo.daily_prices(t)
    def earnings(self, t):          return demo.earnings_dates(t)
    def iv_rank(self, t, *a, **k):  return demo.iv_rank(t)
    def chain(self, t, d, lo, hi):  return demo.chain(t, d, lo, hi)
    def opening_range(self, t, s, e): return demo.opening_range(t, s, e)


class AlphaVantageSource:
    name = "Live (Alpha Vantage)"
    synthetic = False

    def prices(self, t):            return av.daily_prices(t)
    def earnings(self, t):          return av.earnings_dates(t)
    def iv_rank(self, t, start=None, end=None, progress=None):
        return av.iv_rank(t, start, end, progress)
    def chain(self, t, d, lo, hi):  return av.chain(t, d, lo, hi)
    def opening_range(self, t, s, e): return av.opening_range(t, s, e)


SOURCES = {"Alpha Vantage": AlphaVantageSource, "ORATS + FMP": LiveSource, "Demo (synthetic)": DemoSource}


def get_source(name):
    """name: a SOURCES key (True/False kept for old callers: True = demo)."""
    if name is True:
        return DemoSource()
    if name is False:
        return LiveSource()
    return SOURCES[name]()
