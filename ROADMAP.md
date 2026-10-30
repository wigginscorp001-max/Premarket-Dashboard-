# Roadmap – Wiggins Investments LLC

## DATA POLICY (locked 29/Oct/2026)
- No free/scraped data sources (no Yahoo Finance, Wikipedia, iShares downloads, etc.).
- Unusual Whales is the primary feed. If UW can't provide something, tell Sa'Rah El and recommend
  the right paid subscription (stocks, options, futures, forex, crypto) for the site and app.

## Website / app – planned
- **Scanner** (BUILT 29/Oct/2026, 🔎 Scanner tab) on the page and the app, built on Unusual Whales data:
  gap %, relative volume, pre-market high/low breaks, opening-range breakouts,
  unusual options flow; watchlist + whole sectors.
- **Confluence check (BUILT 29/Oct/2026, 🎯 Confluence tab; quick, phone-friendly):** one card per ticker showing which signals agree
  (trend vs EMA 9/200, VWAP, MACD, gap, relative volume, PM high/low, opening-range break,
  options flow, nearest support/resistance) with a simple score, e.g. "6 of 8 bullish".
  Used as a second opinion while looking at other charts or on the go. Short plain-English hints, not a full guide.
- **Heat maps (BUILT 29/Oct/2026, 🟩 Heat Map tab; also Crypto L1/L2), Robinhood style:** tiles sized by market cap (bigger company = bigger tile
  and bigger ticker text), green/red shading by % change, ticker + % printed inside; tap a tile for its chart.
  Option to size by today's dollar volume instead. Includes: market/sector map,
  watchlist heat map, sector strength, and options-flow heat map (calls vs puts premium).
- **Premium screener** (paid tier) – later, with a commercially licensed data source.
- Finviz/Webull: personal research only (no redistribution); not used inside the site.

## Other projects (in order)
1. Pine Script for TradingView
2. thinkorswim / Schwab API + thinkScript
3. AI trading agent: rules decide, AI advises, hard risk limits, paper trading first
4. Beginner study guide

## Before monetizing
- Commercial data licenses (Unusual Whales commercial terms; replace Yahoo with Polygon/Alpaca/Tradier)
- Lawyer review: "educational, not advice"; past results ≠ future; crypto wording
- Terms of Service + Privacy Policy
- Private repo, paid hosting, logins + Stripe
- Keep TradingView chart credit/link

## Confluence alerts
- Alert when several signals agree on a ticker (default: 4 of 5, user-adjustable):
  gap % ≥ threshold, relative volume ≥ 2x, holding above PM high / ORB high (or below for shorts),
  news catalyst, bullish/bearish UW dark-pool + options flow.
- Delivery: on the site (banner + sound while open) and push to phone (ntfy or Telegram)
  from a scheduled job (Mac or GitHub Actions) during 4:00–11:00 ET, every few minutes.
- One alert per ticker per setup per day (no spam). Alerts only – never places orders.

## Gamma / dealer-hedging layer (from Freddy Sami interview review, 29/Oct/2026)
- Daily GEX levels from UW greek-exposure (SPY, QQQ + watchlist): call wall, put wall,
  gamma flip, regime (positive = pin/mean-revert, negative = trend/bigger moves).
- Draw them as chart levels; add "Gamma regime" + "near call/put wall" to confluence cards.
- OpEx calendar flag (monthly 3rd Friday; quad witching in Mar/Jun/Sep/Dec): pinning into it, release after.
- Focus window: first 1–2 hours; note the 3:50 PM closing-imbalance window.
- Verified 29/Oct/2026 (HTTP 200, SPY, 491 strikes each):
  - /api/stock/{t}/greek-exposure/strike – daily (prior-close OI): date, strike, call_gex, put_gex,
    call_delta, put_delta, call_vanna, put_vanna, call_charm, put_charm.
  - /api/stock/{t}/spot-exposures/strike – intraday, has price + time: call/put_gamma_oi, _vol, _bid, _ask,
    plus delta/vanna/charm in the same four flavors. Use *_gamma_oi for walls, flip and regime.

## Nasdaq-100 & Russell 2000 (BUILT 29/Oct/2026)
- Heat Map + Scanner can load the Nasdaq-100 (Wikipedia list) and the biggest 50–200 Russell 2000
  companies by IWM weight (Unusual Whales ETF holdings if the plan includes them, else iShares download).
- Pre-Market cards add live index ETFs: SPY, QQQ, IWM, DIA, SMH.

## UW coverage check (29/Oct/2026) – all HTTP 200 on Sa'Rah's plan
- stock info (sector, marketcap), daily candles (755 rows), earnings (report_date/time, expected move),
  option contracts (greeks, OI, volume), ETF holdings QQQ (102) – ready to replace Yahoo/Wikipedia.
- SPY holdings returned only 250 rows (not ~503) – check paging before using it for the S&P 500 list.
- Not covered by UW: futures, VIX/10Y/DXY, crypto, forex → paid feeds when Sa'Rah decides.

## Friends beta (after this iteration)
- Small group of friends tests the site; code stays private (owner-only access).
- Steps: make GitHub repo private → give Streamlit access to private repos → app sharing set to
  "only specific people" → invite friends by email (viewer only; no Manage app, no Secrets, no code).
- Check data licenses before sharing (UW / FMP personal plans may not allow showing data to others).

## Built 30/Oct/2026 (overnight)
- 🐳 Ticker Intel tab: UW profile, IV rank, max pain, gamma, options volume, flow alerts, net premium,
  dark pool, OI change, volume by price, IV term structure, short interest, insiders, Congress,
  analysts, institutions, earnings history, seasonality, news (paths to confirm with uw_check).
- Market mood gauge (fear ↔ greed) on Pre-Market: market tide, SPY put/call, IV rank, gamma,
  momentum, 52-week position, sector breadth.
- 52-week highs & lows table on Pre-Market (UW daily candles).
