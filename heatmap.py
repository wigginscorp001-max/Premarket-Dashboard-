"""Robinhood-style heat maps: one tile per ticker, sized by company value (or dollar
volume / options premium) and shaded green-to-red by today's move.

Text size follows tile size, so bigger companies get bigger tickers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

RED, FLAT, GREEN = "#c0262d", "#3a3f4b", "#16a34a"
SCALE = [[0.0, RED], [0.5, FLAT], [1.0, GREEN]]

# SPDR sector ETFs for the sector-strength map
SECTOR_ETFS = {
    "XLK": "Technology", "XLC": "Communication", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care",
    "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities",
}


def usd(v: float) -> str:
    return "—" if not v == v else f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def money(v: float) -> str:
    if not v == v:
        return "—"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v / div:,.1f}{unit}"
    return f"${v:,.0f}"


def treemap(df: pd.DataFrame, *, size: str, color: str, text: list[str], hover: list[str],
            group: str | None = None, color_range: float = 3.0, height: int = 620,
            center: float = 0.0) -> go.Figure:
    """Tiles from rows of df. `size` and `color` name columns; `text`/`hover` are per-row strings.

    With `group`, tiles are nested in group boxes (tap a box to zoom in, tap the bar on top to go back).
    Colour is clipped to center ± color_range, so a 3% move is already full green/red.
    """
    d = df.assign(_text=text, _hover=hover).copy()
    d = d[d[size].notna() & (d[size] > 0)]
    if d.empty:
        return go.Figure()
    ids, labels, parents, values, colors, texts, hovers = [], [], [], [], [], [], []
    if group:
        for g, part in d.groupby(group, sort=False):
            w = part[size] / part[size].sum()
            avg = float((part[color].fillna(center) * w).sum())
            ids.append(f"g:{g}"); labels.append(str(g)); parents.append("")
            values.append(float(part[size].sum())); colors.append(avg)
            texts.append(f"<b>{g}</b>"); hovers.append(f"<b>{g}</b><br>{len(part)} tickers · weighted {avg:+.2f}")
    for _, r in d.iterrows():
        ids.append(f"t:{r['Ticker']}"); labels.append(r["Ticker"])
        parents.append(f"g:{r[group]}" if group else "")
        values.append(float(r[size])); colors.append(float(r[color]) if r[color] == r[color] else center)
        texts.append(r["_text"]); hovers.append(r["_hover"])
    # Robinhood look: text grows with the tile. Font from the tile's share of the whole map
    # (Plotly still shrinks anything that doesn't fit).
    total = float(d[size].sum())
    fonts = [int(np.clip(10 + 70 * np.sqrt(v / total), 10, 46)) for v in values]
    if group:   # group headers stay small
        fonts = [13 if i.startswith("g:") else f for i, f in zip(ids, fonts)]
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        text=texts, texttemplate="%{text}", hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(colors=colors, colorscale=SCALE, cmin=center - color_range, cmax=center + color_range,
                    line=dict(width=1.5, color="#0b0f17"), pad=dict(t=22 if group else 2, l=2, r=2, b=2)),
        textfont=dict(size=fonts, color="#ffffff"), textposition="middle center",
        pathbar=dict(visible=bool(group), thickness=22), tiling=dict(pad=2), maxdepth=3,
    ))
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def move_tiles(df: pd.DataFrame, unit: str = "share") -> tuple[list[str], list[str]]:
    """Tile text and hover text for a price-move map: ticker, % change and $ change per share (or coin)."""
    short = {"share": "sh", "coin": "coin"}.get(unit, unit)
    text, hover = [], []
    for _, r in df.iterrows():
        pct = r["Change %"]
        p = f"{pct:+.2f}%" if pct == pct else "—"
        text.append(f"<b>{r['Ticker']}</b><br>{p}<br>{usd(r['Change $'])}/{short}")
        hover.append(f"<b>{r['Ticker']}</b>" + (f" · {r['Name']}" if r.get("Name") else "")
                     + f"<br>Price {r['Price']:,.2f}<br>Change {p} ({usd(r['Change $'])} per {unit})"
                     + f"<br>Market cap {money(r.get('Market cap', np.nan))}")
    return text, hover


def flow_tiles(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Tile text and hover text for the options-flow map."""
    text, hover = [], []
    for _, r in df.iterrows():
        cp = r["Calls %"]
        side = f"{cp:.0f}% calls" if cp >= 50 else f"{100 - cp:.0f}% puts"
        text.append(f"<b>{r['Ticker']}</b><br>{side}<br>{money(r['Premium'])}")
        hover.append(f"<b>{r['Ticker']}</b><br>Calls {money(r['Calls $'])} · Puts {money(r['Puts $'])}"
                     f"<br>{int(r['Alerts'])} unusual alerts")
    return text, hover
