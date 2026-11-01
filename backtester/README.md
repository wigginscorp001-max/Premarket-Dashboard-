# Wiggins Investments LLC — Options Backtester (v1)

Private, backtest-only app. Separate from the live dashboard.

**v1 strategies:** 45–60 DTE call/put swings (delta 0.60–0.70) and LEAPS (1–2 yrs, delta 0.70–0.85).
Spreads come later.

## How a backtest works
1. Each day, entry rules are checked on the **close** (FMP daily prices).
2. If they're all true, the trade is opened the **next trading day** on the ORATS snapshot:
   **buy at the ask**, + commission. Contract = closest to target delta inside your DTE range,
   with a max bid/ask spread and min open interest.
3. Every day after, the contract is priced from that day's ORATS chain. It's closed (**sold at the bid**,
   + commission) at the first of: stop loss · profit target · time stop (DTE) · earnings exit ·
   signal reversal · max days · expiry.
4. Equity is marked to market daily (mid price) for the equity curve and drawdown.

Limits to know: daily data can't see intraday stop hits, so stops fill at that day's bid (often worse
than the stop — conservative). Only one open position per ticker.

## Entry rules (tick any; all ticked must be true)
Trend pullback · Breakout (N-day high/low) · EMA 9/200 · MACD 12/26/9 cross · IV-rank filter (cheap IV) ·
Gap % · Relative volume · 30-min opening range 9:30–10:00 (needs Polygon key) · Earnings: skip / hold / exit before.

## Results
Win rate, avg win/loss, expectancy, profit factor, total return, CAGR, max drawdown, equity curve,
breakdowns by year / ticker / market regime (SPY above/below 200-day) / IV regime (IV rank ≥ 50) /
calls vs puts / exit reason, every trade (CSV download), a chart of any trade, and **🤖 AI Analysis** (Claude).

## Files
| File | What it does |
|---|---|
| `app.py` | The Streamlit app (pages: Setup, Results, Trades, AI Analysis) |
| `bt/engine.py` | Day-by-day backtest engine |
| `bt/signals.py`, `bt/indicators.py` | Entry / exit rules, EMA, MACD, ATR, RVOL, gaps |
| `bt/orats.py` | ORATS option chains + IV rank |
| `bt/fmp.py` | FMP prices + earnings dates |
| `bt/polygon.py` | Optional 30-min opening range |
| `bt/store.py` | Saves every download (Supabase/Postgres or local SQLite) so nothing is re-downloaded |
| `bt/metrics.py`, `bt/charts.py` | Stats and charts |
| `bt/ai.py` | Claude analysis |
| `bt/demo.py` | Synthetic data for testing the app only |

## Secrets (Streamlit Cloud → app → Settings → Secrets)
```
ORATS_API_KEY = "..."
FMP_API_KEY = "..."
ANTHROPIC_API_KEY = "..."
UW_API_KEY = "..."
# optional
POLYGON_API_KEY = "..."
DATABASE_URL = "postgresql://..."   # Supabase connection string (Session pooler)
```
Without `DATABASE_URL`, downloads are cached in a local file that Streamlit Cloud wipes on reboot.
Add Supabase to keep them permanently.

## Run on the iMac
```bash
pip install -r requirements.txt
streamlit run app.py
```
Put keys in `.streamlit/secrets.toml` (same format as above; this file is git-ignored).

## Tests
`python -m pytest tests` (uses demo data, no keys needed).
