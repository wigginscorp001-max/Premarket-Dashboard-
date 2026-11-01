"""Plotly charts – dark, minimal, phone friendly. Dates shown as dd/Mon/yyyy."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN = "#0ca30c", "#d03b3b"
BLUE, MUTED, GRID = "#2a78d6", "#898781", "rgba(148,163,184,.12)"
DATE = "%d/%b/%Y"


def _style(fig, height):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=8, r=8, t=30, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=12, color="#cbd5e1"),
                      hoverlabel=dict(bgcolor="#111827", bordercolor="#1f2937"),
                      showlegend=False, bargap=0.25)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return fig


def equity(eq: pd.DataFrame, capital: float, trades: pd.DataFrame | None = None):
    dd = (eq["equity"] / eq["equity"].cummax() - 1) * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.06, subplot_titles=("Equity", "Drawdown %"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq["equity"], mode="lines", line=dict(color=BLUE, width=2),
                             hovertemplate=f"%{{x|{DATE}}}<br>Equity $%{{y:,.0f}}<extra></extra>"), 1, 1)
    fig.add_hline(y=capital, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", fill="tozeroy",
                             line=dict(color=DOWN, width=1.5), fillcolor="rgba(208,59,59,.18)",
                             hovertemplate=f"%{{x|{DATE}}}<br>Drawdown %{{y:.1f}}%<extra></extra>"), 2, 1)
    fig.update_xaxes(tickformat=DATE, hoverformat=DATE)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return _style(fig, 440)


def pnl_bars(df: pd.DataFrame, col: str = "Total P&L $", title: str = ""):
    if df.empty:
        return None
    x = [str(i) for i in df.index]
    y = df[col]
    fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=[UP if v > 0 else DOWN for v in y],
                                                 cornerradius=4),
                           customdata=df[["Trades", "Win %"]].values,
                           hovertemplate="%{x}<br>P&L $%{y:,.0f}<br>%{customdata[0]} trades · "
                                         "win %{customdata[1]:.0f}%<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(type="category")
    fig.update_layout(title=dict(text=title, font=dict(size=13)))
    return _style(fig, 280)


def pnl_hist(tr: pd.DataFrame):
    if tr.empty:
        return None
    fig = go.Figure()
    for won, color in ((True, UP), (False, DOWN)):
        s = tr.loc[tr["win"] == won, "pnl_pct"]
        fig.add_trace(go.Histogram(x=s, marker_color=color, opacity=0.85, xbins=dict(size=10),
                                   name="Wins" if won else "Losses",
                                   hovertemplate="%{x}%<br>%{y} trades<extra></extra>"))
    fig.update_layout(barmode="overlay", title=dict(text="Trade return % (green = wins, red = losses)",
                                                    font=dict(size=13)))
    fig.update_xaxes(ticksuffix="%")
    return _style(fig, 260)


def trade_chart(px: pd.DataFrame, trade: pd.Series):
    """Stock price around one trade with entry/exit markers."""
    a = trade["entry_date"] - pd.Timedelta(days=90)
    b = trade["exit_date"] + pd.Timedelta(days=20)
    s = px.loc[a:b]
    fig = go.Figure(go.Candlestick(x=s.index, open=s["open"], high=s["high"], low=s["low"], close=s["close"],
                                   increasing_line_color=UP, decreasing_line_color=DOWN))
    for d, label, col in ((trade["entry_date"], "Entry", BLUE), (trade["exit_date"], "Exit", MUTED)):
        if d in s.index:
            fig.add_trace(go.Scatter(x=[d], y=[s.at[d, "close"]], mode="markers+text", text=[label],
                                     textposition="top center", marker=dict(size=11, color=col,
                                     line=dict(color="#0b1220", width=2))))
    fig.add_hline(y=trade["strike"], line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text=f"Strike {trade['strike']:g}", annotation_font_color=MUTED)
    fig.update_xaxes(tickformat=DATE, rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    return _style(fig, 340)
