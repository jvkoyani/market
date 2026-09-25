"""Market data access.

Two backends behind one interface:

  * ``yfinance``  -- live OHLCV from Yahoo Finance (run on your own machine).
  * ``sample``    -- Parquet/CSV cached under ``sample_data/`` so the dashboard,
                     tests and CI run fully offline with no network.

Everything downstream (indicators, signals, screener) consumes the normalised
frame this module returns: a ``DatetimeIndex`` and lowercase columns
``open, high, low, close, volume``.
"""
from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["close"])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


# --- sample backend -------------------------------------------------------

def _sample_path(symbol: str) -> str:
    safe = symbol.replace(".", "_")
    return os.path.join(SAMPLE_DIR, f"{safe}.csv")


def load_sample(symbol: str) -> pd.DataFrame | None:
    path = _sample_path(symbol)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalise(df)


def available_sample_symbols() -> list[str]:
    if not os.path.isdir(SAMPLE_DIR):
        return []
    out = []
    for f in sorted(os.listdir(SAMPLE_DIR)):
        if f.endswith(".csv"):
            out.append(f[:-4].replace("_", "."))
    return out


# --- live backend ---------------------------------------------------------

def load_yfinance(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            "yfinance not installed. `pip install yfinance` or use source='sample'."
        ) from e
    try:
        df = yf.download(
            symbol, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _normalise(df)
    except Exception:
        return None


# --- unified entry point --------------------------------------------------

def get_history(symbol: str, source: str = "yfinance", **kwargs) -> pd.DataFrame | None:
    """Fetch OHLCV history for one symbol.

    source: 'yfinance' (live) or 'sample' (offline cache).
    """
    if source == "sample":
        return load_sample(symbol)
    return load_yfinance(symbol, **kwargs)


@lru_cache(maxsize=2048)
def _cached(symbol: str, source: str, period: str, interval: str):
    return get_history(symbol, source=source, period=period, interval=interval)


def get_history_cached(symbol: str, source: str = "yfinance",
                       period: str = "2y", interval: str = "1d") -> pd.DataFrame | None:
    """Process-lifetime cached variant to avoid refetching within a session."""
    return _cached(symbol, source, period, interval)
