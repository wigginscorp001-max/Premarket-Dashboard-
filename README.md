# Trading Command Center

One page for your morning prep: futures, pre-market gaps, earnings, swing setups
with probabilities, LEAPS screening, day-trading levels, and your own scripts.

## Start

```bash
./run.sh                 # macOS / Linux (sets up a venv the first time)
# or
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Edit `watchlist.txt` (or the sidebar box) to change tickers.
Flip **Demo data** in the sidebar to test the layout without internet.

## Pages

| Page | What it shows |
|---|---|
| 🌅 Pre-Market | Futures & macro · every watchlist ticker: prior close, PM last, gap, PM high/low/VWAP, **PM bias**, PM volume |
| 🧭 PM → Close | Today: PM bias → opening range (9:30–10:00) → behavior → afternoon (1–4 PM) · 60-session history per ticker: how often the PM bias continued, chopped, took PM high/low, and whether the afternoon followed |
| 📐 Setups & Patterns | Bull/bear flags, rising/falling wedges, ascending/descending/symmetrical triangles, channels, double tops/bottoms, 20-day breakouts, RSI extremes, inside bars, NR7 · daily or 5-minute · how often each pattern worked on that ticker before |
| 🐋 Options Flow | Unusual Whales: flow alerts, per-ticker options (IV rank, max pain, volume, flow), market tide, dark pool, gamma exposure, earnings today |
| 🎯 Options & LEAPS | Long-dated options with delta, breakeven, P(ITM), P(profit), fit score |
| 📅 Earnings | Upcoming reports, implied vs historical move |
| 📈 Swing Odds | ATR stop/target with historical win rate per setup |
| ⚡ Day Trading | Live chart with VWAP, PM high/low, opening range, pivots, RVOL |
| 🧩 My Scripts | Your own Python/terminal scripts |

## Put it online (Streamlit Community Cloud, free)

1. Push this folder to a **private** GitHub repo.
2. Go to **share.streamlit.io**, sign in with GitHub, click **Create app**, pick the repo,
   branch `main`, main file `app.py`. Under **Advanced settings** choose Python 3.12.
3. In **Advanced settings → Secrets** paste:  `UW_API_KEY = "your-key"`
4. Deploy. Then in the app's **Settings → Sharing**, make it private (only your email).

The watchlist is `watchlist.txt` – edit it on GitHub and the site picks it up.

## Unusual Whales (real-time flow)

1. Copy `.env.example` to `.env` in this folder and put your key after the `=`:
   `UW_API_KEY=your_key_here`
2. Restart the app and open the **🐋 Unusual Whales** tab: flow alerts, market tide, dark pool,
   gamma exposure by strike, today's earnings, and an "Any endpoint" box for everything else
   in UW's API docs.
3. Turn on **Live mode → Auto-refresh** in the sidebar to re-pull flow and pre-market quotes every 10–120 s.

`.env` is listed in `.gitignore`. Never commit it or share it.

## Plug in your own code

Put a `.py` file in `plugins/`:

```python
TITLE = "My scanner"

def run(tickers):          # gets the watchlist
    ...                    # paste your code here
    return df              # DataFrame, dict, text, plotly Figure, or a list of them
```

Already have a script you run in the terminal (any language)? Copy
`plugins/example_terminal_script.py` and set `COMMAND = "python ~/my_scan.py {tickers}"`.
Its printed output appears in the dashboard.

## Notes

* Data comes from Yahoo Finance through `yfinance`. It's free and can be delayed. To use a paid
  feed (Polygon, Alpaca, Schwab, IBKR), rewrite the functions in `data.py` and keep the same return shapes.
* Thin ETFs (XLK, XLF…) can go a long time without a pre-market trade. Those rows say
  *"no PM trades yet"* instead of disappearing.
* Probabilities are statistical estimates (lognormal / random walk / historical frequency),
  not predictions. Not financial advice.
* Add your own swing setups in `analytics.setup_signals()`. They get backtested and appear in the scan.
