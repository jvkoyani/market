"""Generate synthetic OHLCV sample data for offline mode / tests / CI.

This produces *plausible* price series (geometric random walk with regime
drift and volatility clustering) -- NOT real market data. It exists so the
dashboard and test suite run with zero network access. For real signals,
run the app with ``source='yfinance'`` on a machine with internet.

Usage:  python generate_sample_data.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from stockdash.universe import NIFTY_50, to_yahoo

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

# A few named regimes so the screener yields a mix of BUY/HOLD/SELL.
REGIMES = {
    "uptrend": dict(drift=0.0009, vol=0.014),
    "strong_uptrend": dict(drift=0.0016, vol=0.018),
    "downtrend": dict(drift=-0.0011, vol=0.017),
    "sideways": dict(drift=0.0001, vol=0.012),
    "choppy": dict(drift=0.0003, vol=0.024),
}


def _gen_series(n: int, start: float, drift: float, vol: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Volatility clustering via a slow-moving scale factor.
    scale = 1 + 0.5 * np.sin(np.linspace(0, 6 * np.pi, n)) ** 2
    rets = rng.normal(drift, vol, n) * scale
    close = start * np.exp(np.cumsum(rets))
    # Build OHLC around the close path.
    intraday = np.abs(rng.normal(0, vol, n))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, vol / 3, n))
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    base_vol = rng.integers(3_00_000, 30_00_000)
    volume = (base_vol * (1 + 0.6 * np.abs(rets) / vol)).astype(int)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def main(days: int = 600):
    os.makedirs(OUT_DIR, exist_ok=True)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    regime_names = list(REGIMES)
    symbols = to_yahoo(NIFTY_50)

    for i, sym in enumerate(symbols):
        regime = REGIMES[regime_names[i % len(regime_names)]]
        start_price = 200 + (i * 137) % 3800
        df = _gen_series(days, start_price, regime["drift"], regime["vol"], seed=1000 + i)
        df.index = dates
        df.index.name = "date"
        safe = sym.replace(".", "_")
        df.to_csv(os.path.join(OUT_DIR, f"{safe}.csv"))

    # A NIFTY index proxy for the market-regime filter (mild uptrend).
    idx = _gen_series(days, 22000, 0.0006, 0.010, seed=42)
    idx.index = dates
    idx.index.name = "date"
    idx.to_csv(os.path.join(OUT_DIR, "_NSEI.csv"))

    print(f"Wrote {len(symbols)} symbols + index to {OUT_DIR}")


if __name__ == "__main__":
    main()
