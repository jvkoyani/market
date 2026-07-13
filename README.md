# 📈 Indian Stock Market — Signal Dashboard

A dashboard that screens the NSE universe with multi-factor technical analysis
and produces actionable **BUY / SELL / HOLD** calls — each with an **entry
price, target, and stop-loss** — then sizes a **risk-managed portfolio** whose
whole design goal is to finish the quarter *positive*.

> ⚠️ **Read this first.** No tool can *guarantee* a green portfolio every
> quarter — markets are uncertain and anyone promising certainty is wrong.
> What this system does is stack the odds toward ending positive through
> disciplined **risk management**: small per-trade risk, favourable
> reward:risk, a market-regime cash filter, and diversification. It is an
> **educational, algorithmic analysis tool — not investment advice.** Consult a
> SEBI-registered advisor before investing real money.

---

## Why this design gives the best shot at a positive quarter

Returns can't be promised, but losses *can* be controlled — and controlling
losses is what keeps a quarter green. Five rules do the heavy lifting:

1. **Fixed-fractional risk (≤1% default).** Each trade risks a tiny slice of
   capital. Even a run of losers barely dents the book.
2. **Reward:risk ≥ 2:1 enforced.** Targets/stops come from volatility (ATR).
   Any setup that can't offer ≥2× reward for its risk is downgraded to HOLD —
   a skipped trade never loses money.
3. **Market-regime filter.** When the Nifty is below its 200-EMA (RISK-OFF),
   max exposure is throttled toward cash (~25%). You sit out the worst tape.
4. **Diversification caps.** No single position exceeds ~15% of the book;
   total exposure is capped (default 80%), always keeping a cash buffer.
5. **Backtested mechanics.** The exact signal + risk rules are replayed
   bar-by-bar so you can see the edge and drawdowns before risking capital.

The math that matters: risk 1% per trade at 2:1 reward with a ~40% win rate is
net-positive over a quarter's worth of trades, *and* the worst-case loss if
every open stop triggers at once is capped at a few percent of capital.

---

## Quick start

```bash
pip install -r requirements.txt

# Offline demo (synthetic data, no internet needed):
streamlit run app.py            # pick "Offline sample" in the sidebar

# Live NSE data:
streamlit run app.py            # pick "Live NSE (yfinance)" in the sidebar
```

Then open the URL Streamlit prints (usually http://localhost:8501).

### Data modes
- **Live NSE (yfinance):** real OHLCV from Yahoo Finance (`RELIANCE.NS`, …).
  Needs internet. Choose Nifty 50 / 100 / or a live Nifty 500 download.
- **Offline sample:** bundled **synthetic** data in `sample_data/`. Lets you
  explore the whole app with zero network. *Not real prices* — for demo/testing
  only. Regenerate with `python generate_sample_data.py`.

---

## What each tab shows

| Tab | What you get |
|-----|--------------|
| 🔎 **Signals** | Ranked table of every stock: action, entry, target, stop-loss, reward:risk, conviction score, RSI, ADX, and a plain-English reason. Downloadable as CSV. |
| 💼 **Suggested Portfolio** | Today's BUYs turned into a sized, risk-capped book: shares, capital, rupee risk per name, cash buffer, and worst-case total risk. |
| 🔬 **Stock Detail** | Price with EMA 20/50/200, RSI, and MACD histogram for any symbol, plus its signal breakdown. |
| 🧪 **Backtest** | Replays the strategy over history — win rate, total return, profit factor, and an equity curve. |

---

## How a signal is built

For every stock we compute a standard indicator set (EMA 20/50/200, RSI, MACD,
ATR, ADX, Bollinger, ROC, volume average, 20-day & 52-week ranges) and score it
0–100 across four buckets:

- **Trend structure (35 pts)** — price vs EMAs, EMA alignment.
- **Momentum (30 pts)** — RSI zone, MACD, rate-of-change.
- **Trend strength (15 pts)** — ADX.
- **Confirmation (20 pts)** — above-average volume, 20-day breakout.

A **BUY** needs a high score, a non-down trend, minimum trend strength (ADX),
*and* a ≥2:1 reward:risk. Otherwise it's **HOLD**. Weak/declining names are
**SELL / avoid**.

- **Target** = entry + `N × ATR` (default 3×)
- **Stop-loss** = entry − `M × ATR` (default 1.5×)

Everything (thresholds, ATR multiples, risk %, exposure caps) is tunable from
the sidebar.

---

## Project layout

```
market/
├── app.py                     # Streamlit dashboard (entry point)
├── generate_sample_data.py    # builds offline synthetic data
├── requirements.txt
├── stockdash/
│   ├── universe.py            # NSE ticker lists (Nifty 50 / broad / live 500)
│   ├── data.py                # OHLCV fetch: yfinance (live) + sample (offline)
│   ├── indicators.py          # pure technical-indicator functions
│   ├── signals.py             # BUY/SELL/HOLD + entry/target/stop + score
│   ├── risk.py                # position sizing, exposure caps, regime filter
│   ├── screener.py            # run pipeline across a universe -> ranked table
│   └── backtest.py            # bar-by-bar validation of the rules
├── sample_data/               # bundled synthetic OHLCV (offline mode)
└── tests/                     # pytest suite (indicators, signals, risk)
```

## Running the tests

```bash
pytest -q
```

## Using the engine without the dashboard

```python
from stockdash import screener, data, risk

syms = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
table = screener.screen(syms, source="yfinance")     # ranked signals
buys  = screener.top_buys(table)
book  = risk.build_portfolio(1_000_000, buys)         # sized, risk-capped
print(book["positions"])
```

---

## Disclaimer

This software is provided for **education and research**. Algorithmic signals
are derived from historical price patterns and **do not guarantee future
returns**. Trading and investing carry risk of loss. Nothing here is a
recommendation to buy or sell any security. Do your own research and consult a
SEBI-registered financial advisor.
