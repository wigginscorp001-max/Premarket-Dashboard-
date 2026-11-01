"""Claude explains backtest results and suggests rule changes (streams into the page)."""
from __future__ import annotations

import json

import pandas as pd

from .config import AI_MODEL, secret
from . import metrics as mx

SYSTEM = """You are the quantitative analyst for Wiggins Investments LLC, reviewing a backtest of a
long-options strategy (calls/puts, 45–60 DTE swings or 1–2 year LEAPS). Fills were at historical
bid/ask plus commissions, with signals on day t close and fills on day t+1 (no look-ahead).

Write for a busy trader reading on a phone:
- Start with a 2–3 sentence verdict: is there an edge, and how confident should we be?
- Then sections: What worked · What hurt · Rule changes to test next · Risks / caveats.
- Every claim must cite numbers from the data provided. Never invent numbers.
- Rule changes must be concrete and testable in this tool (e.g. "raise IV-rank max from 30 to 40",
  "tighten stop from 40% to 30%", "puts only in Bear regime").
- Flag small samples (under ~30 trades in a bucket) as unreliable, and flag overfitting risk.
- Dates use dd/Mon/yyyy. Keep it under ~450 words unless asked for more."""


def _table(df: pd.DataFrame) -> str:
    return df.round(2).to_csv() if df is not None and not df.empty else "(none)"


def build_context(res, summ: dict) -> str:
    p, tr = res.params, res.trades
    settings = {k: (v if not isinstance(v, pd.Timestamp) else v.strftime("%d/%b/%Y"))
                for k, v in vars(p).items() if k != "rules"}
    settings["entry_rules"] = {k: v for k, v in vars(p.rules).items()}
    parts = [
        "SETTINGS:\n" + json.dumps(settings, default=str, indent=1),
        "SUMMARY:\n" + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summ.items()},
                                  default=str),
        "SKIPPED SIGNALS:\n" + json.dumps(res.skipped),
    ]
    if not tr.empty:
        for by, name in (("year", "BY YEAR"), ("ticker", "BY TICKER"), ("market", "BY MARKET REGIME"),
                         ("iv_regime", "BY IV REGIME"), ("type", "CALLS vs PUTS"), ("exit_reason", "BY EXIT REASON")):
            parts.append(f"{name}:\n{_table(mx.breakdown(tr, by))}")
        cols = ["ticker", "type", "entry_date", "exit_date", "dte_entry", "delta_entry", "entry_px",
                "exit_px", "pnl", "pnl_pct", "days_held", "exit_reason", "market", "iv_regime", "ivr_entry"]
        sample = tr[cols].copy()
        for c in ("entry_date", "exit_date"):
            sample[c] = sample[c].dt.strftime("%d/%b/%Y")
        if len(sample) > 150:
            sample = pd.concat([sample.nlargest(40, "pnl"), sample.nsmallest(40, "pnl"),
                                sample.sample(70, random_state=1)]).drop_duplicates()
            parts.append("TRADES (sample: 40 best, 40 worst, 70 random):")
        else:
            parts.append("ALL TRADES:")
        parts.append(sample.round(3).to_csv(index=False))
    if res.notes:
        parts.append("DATA NOTES:\n" + "\n".join(res.notes[:20]))
    return "\n\n".join(parts)


def stream_analysis(context: str, question: str | None = None):
    """Generator of text chunks (for st.write_stream)."""
    import anthropic

    key = secret("ANTHROPIC_API_KEY")
    if not key:
        yield "ANTHROPIC_API_KEY is missing from Streamlit Secrets."
        return
    client = anthropic.Anthropic(api_key=key)
    ask = question or "Explain these results and suggest the next rule changes to test."
    msgs = [{"role": "user", "content": f"{context}\n\nREQUEST: {ask}"}]
    body = {"output_config": {"effort": "high"}}

    def _run(with_fallback: bool):
        extra = dict(body, fallbacks="default") if with_fallback else body
        kw = dict(model=AI_MODEL, max_tokens=8000, system=SYSTEM, messages=msgs, extra_body=extra)
        if with_fallback:
            return client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], **kw)
        return client.messages.stream(**kw)

    try:
        try:
            cm = _run(True)
            stream = cm.__enter__()
        except anthropic.BadRequestError:
            cm = _run(False)
            stream = cm.__enter__()
        try:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
            if getattr(final, "stop_reason", None) == "refusal":
                yield "\n\n_(Claude declined part of this request.)_"
        finally:
            cm.__exit__(None, None, None)
    except anthropic.AuthenticationError:
        yield "\n\nAnthropic rejected the API key – check ANTHROPIC_API_KEY in Secrets."
    except anthropic.RateLimitError:
        yield "\n\nAnthropic rate limit hit – wait a minute and try again."
    except anthropic.APIStatusError as e:
        yield f"\n\nAnthropic API error ({e.status_code}): {e.message}"
    except anthropic.APIConnectionError:
        yield "\n\nCould not reach the Anthropic API (network)."
