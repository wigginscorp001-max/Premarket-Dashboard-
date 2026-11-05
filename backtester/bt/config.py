"""Reads API keys from Streamlit Secrets (or environment variables when run locally).

Keys are NEVER stored in code. Names used:
  ALPHAVANTAGE_API_KEY (or ORATS_API_KEY + FMP_API_KEY), ANTHROPIC_API_KEY,
  ALPHAVANTAGE_RPM (optional, default 75 = requests/minute on your plan),
  POLYGON_API_KEY (optional, 30-min opening range), DATABASE_URL (optional, Supabase/Postgres)
"""
from __future__ import annotations

import os

DATE_FMT = "%d/%b/%Y"          # 05/Jan/2024
AI_MODEL = "claude-opus-5-5"

KEY_NAMES = ["ALPHAVANTAGE_API_KEY", "ANTHROPIC_API_KEY", "ORATS_API_KEY", "FMP_API_KEY",
             "POLYGON_API_KEY", "DATABASE_URL"]


def secret(name: str) -> str | None:
    val = None
    try:
        import streamlit as st
        val = st.secrets.get(name)
    except Exception:          # no secrets file / not running under Streamlit
        val = None
    val = val or os.environ.get(name)
    return str(val).strip() if val else None


def key_status() -> dict[str, bool]:
    return {k: bool(secret(k)) for k in KEY_NAMES}


def fmt_date(d) -> str:
    try:
        return d.strftime(DATE_FMT)
    except Exception:
        return ""
