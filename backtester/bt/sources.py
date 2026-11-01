"""One interface over the paid feeds (FMP + ORATS + optional Polygon) or demo data."""
from __future__ import annotations

import pandas as pd

from . import demo, fmp, orats, polygon


class LiveSource:
    name = "Live (ORATS + FMP)"
    synthetic = False

    def prices(self, t):            return fmp.daily_prices(t)
    def earnings(self, t):          return fmp.earnings_dates(t)
    def iv_rank(self, t):           return orats.iv_rank(t)
    def chain(self, t, d, lo, hi):  return orats.chain(t, d, lo, hi)
    def opening_range(self, t, s, e): return polygon.opening_range(t, s, e)


class DemoSource:
    name = "Demo (synthetic – not real data)"
    synthetic = True

    def prices(self, t):            return demo.daily_prices(t)
    def earnings(self, t):          return demo.earnings_dates(t)
    def iv_rank(self, t):           return demo.iv_rank(t)
    def chain(self, t, d, lo, hi):  return demo.chain(t, d, lo, hi)
    def opening_range(self, t, s, e): return demo.opening_range(t, s, e)


def get_source(demo_mode: bool):
    return DemoSource() if demo_mode else LiveSource()
