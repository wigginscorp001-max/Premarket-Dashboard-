"""Performance statistics and breakdown tables."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summary(tr: pd.DataFrame, eq: pd.DataFrame, capital: float) -> dict:
    out = {"trades": 0}
    if eq is not None and not eq.empty:
        e = eq["equity"]
        peak = e.cummax()
        dd = e / peak - 1
        years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
        out.update({
            "final_equity": float(e.iloc[-1]),
            "total_return_pct": (e.iloc[-1] / capital - 1) * 100,
            "cagr_pct": ((e.iloc[-1] / capital) ** (1 / years) - 1) * 100 if e.iloc[-1] > 0 else -100.0,
            "max_dd_pct": float(dd.min() * 100),
            "max_dd_usd": float((e - peak).min()),
        })
    if tr is None or tr.empty:
        return out
    w, l = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
    gross_w, gross_l = w["pnl"].sum(), -l["pnl"].sum()
    wr = len(w) / len(tr)
    out.update({
        "trades": len(tr),
        "win_rate": wr * 100,
        "avg_win": w["pnl"].mean() if len(w) else 0.0,
        "avg_loss": l["pnl"].mean() if len(l) else 0.0,
        "avg_win_pct": w["pnl_pct"].mean() if len(w) else 0.0,
        "avg_loss_pct": l["pnl_pct"].mean() if len(l) else 0.0,
        "expectancy": tr["pnl"].mean(),
        "expectancy_pct": tr["pnl_pct"].mean(),
        "profit_factor": gross_w / gross_l if gross_l > 0 else np.inf,
        "total_pnl": tr["pnl"].sum(),
        "avg_days": tr["days_held"].mean(),
        "best": tr["pnl"].max(),
        "worst": tr["pnl"].min(),
        "commissions": tr["commission_total"].sum(),
    })
    return out


def breakdown(tr: pd.DataFrame, by: str) -> pd.DataFrame:
    if tr is None or tr.empty:
        return pd.DataFrame()
    g = tr.groupby(by)
    df = pd.DataFrame({
        "Trades": g.size(),
        "Win %": g["win"].mean() * 100,
        "Avg P&L $": g["pnl"].mean(),
        "Avg P&L %": g["pnl_pct"].mean(),
        "Total P&L $": g["pnl"].sum(),
        "Avg days": g["days_held"].mean(),
    })
    return df.sort_values("Total P&L $", ascending=False) if by != "year" else df.sort_index()


def drawdown(eq: pd.DataFrame) -> pd.Series:
    e = eq["equity"]
    return (e / e.cummax() - 1) * 100
