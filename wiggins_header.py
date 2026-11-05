"""Wiggins Investments LLC – blue header, same look as the Premarket Dashboard.

Use in any Streamlit app (put this file next to that app's app.py):

    import wiggins_header
    wiggins_header.show("OPTIONS DESK", feed_name="SCHWAB + UW", live=True)

Call it right after st.set_page_config(...), before anything else is drawn.
"""
from __future__ import annotations

import datetime as dt

import streamlit as st

COMPANY = "Wiggins Investments LLC"
OPERATOR = "Sa'Rah El"
AUM = "$5 Million"            # Assets Under Management – edit here
VALUATION = "$50 Million"     # Market cap – edit here

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.wig-hdr { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;
  background:linear-gradient(135deg, #1e3a8a 0%, #172554 100%); border:1px solid #2563eb;
  border-radius:14px; padding:14px 18px; margin:0 0 12px 0; box-shadow:0 6px 24px rgba(37,99,235,.25);
  font-family:'Inter', sans-serif; }
.wig-brand { display:flex; align-items:center; gap:12px; }
.wig-logo { background:#2563eb; border-radius:10px; padding:6px 9px; font-size:20px;
  box-shadow:0 4px 14px rgba(59,130,246,.45); }
.wig-name { font-weight:800; letter-spacing:.04em; color:#fff; font-size:18px; }
.wig-badge { font:700 10px ui-monospace, monospace; color:#93c5fd; background:rgba(30,58,138,.8);
  border:1px solid #3b82f6; border-radius:999px; padding:2px 8px; margin-left:6px; vertical-align:middle; }
.wig-sub { color:#cbd5e1; font-size:12px; }
.wig-kpi { color:#fff; font-weight:600; }
.wig-status { display:flex; gap:8px; flex-wrap:wrap; }
.wig-pill { background:#0f172a; border:1px solid rgba(96,165,250,.45); border-radius:8px; padding:5px 10px;
  font:12px ui-monospace, monospace; color:#e2e8f0; }
.wig-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.wig-dot.ok { background:#10b981; animation:wigpulse 1.6s infinite; } .wig-dot.off { background:#f59e0b; }
b.wig-ok { color:#34d399; } b.wig-off { color:#fbbf24; }
@keyframes wigpulse { 50% { opacity:.35; } }
@media (max-width: 640px) { .wig-name { font-size:14px; } .wig-badge { display:none; } }
</style>
"""


def show(badge: str = "OPTIONS DESK", feed_name: str = "DATA", live: bool = True,
         status_text: str | None = None, logo: str = "📈") -> None:
    """Draw the header. live=False shows an orange dot and status_text (default 'OFFLINE')."""
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now = dt.datetime.utcnow()
    state = status_text or ("LIVE" if live else "OFFLINE")
    cls = "ok" if live else "off"
    st.markdown(_CSS + f"""
<div class="wig-hdr">
  <div class="wig-brand">
    <div class="wig-logo">{logo}</div>
    <div>
      <div class="wig-name">{COMPANY.upper()} <span class="wig-badge">{badge}</span></div>
      <div class="wig-sub">Operator: {OPERATOR}</div>
      <div class="wig-sub">Assets Under Management <b class="wig-kpi">{AUM}</b> · Market Cap <b class="wig-kpi">{VALUATION}</b></div>
    </div>
  </div>
  <div class="wig-status">
    <div class="wig-pill"><span class="wig-dot {cls}"></span>{feed_name}: <b class="wig-{cls}">{state}</b></div>
    <div class="wig-pill">NY {now:%a %d/%b/%Y %H:%M}</div>
  </div>
</div>
""", unsafe_allow_html=True)
