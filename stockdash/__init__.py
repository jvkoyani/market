"""stockdash -- Indian stock market analysis & signal engine.

Modules:
    universe    NSE ticker lists (Nifty 50 / broad / live Nifty 500)
    data        OHLCV fetch (yfinance live + offline sample cache)
    indicators  pure technical-indicator functions
    signals     BUY/SELL/HOLD with entry, target, stop-loss, conviction score
    risk        position sizing, exposure caps, market-regime filter
    screener    run the pipeline across a universe -> ranked table
    backtest    validate the signal + risk edge on history
"""
from . import universe, data, indicators, signals, risk, screener, backtest  # noqa: F401

__version__ = "0.1.0"
