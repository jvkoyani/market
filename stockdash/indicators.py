"""Technical indicators.

Every function is pure: it takes an OHLCV ``pandas.DataFrame`` (or a price
Series) and returns indicator values aligned to the same index. No network,
no globals -- this keeps the analysis testable and deterministic.

Expected DataFrame columns (case-insensitive handled by ``data.py``):
    open, high, low, close, volume
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing), 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # If there were no losses, RSI is 100; if no gains, RSI is 0.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, out.where(avg_loss == 0, 0.0))
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing). Basis for stops/targets."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (trend strength, 0-100)."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands -> (middle, upper, lower)."""
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def roc(close: pd.Series, period: int = 20) -> pd.Series:
    """Rate of change (%) over ``period`` bars."""
    return 100 * (close / close.shift(period) - 1)


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).min()


def compute_all(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Attach a standard indicator set to an OHLCV frame.

    Returns a copy of ``df`` with extra columns. Used by the screener and the
    dashboard so every stock is evaluated with the same feature set.
    """
    p = params or {}
    out = df.copy()
    close = out["close"]

    out["ema20"] = ema(close, p.get("ema_fast", 20))
    out["ema50"] = ema(close, p.get("ema_mid", 50))
    out["ema200"] = ema(close, p.get("ema_slow", 200))
    out["rsi"] = rsi(close, p.get("rsi_period", 14))
    macd_line, signal_line, hist = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["atr"] = atr(out, p.get("atr_period", 14))
    out["adx"] = adx(out, p.get("adx_period", 14))
    mid, upper, lower = bollinger(close, p.get("bb_window", 20))
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["roc"] = roc(close, p.get("roc_period", 20))
    out["vol_sma"] = sma(out["volume"], p.get("vol_window", 20))
    out["hi_20"] = rolling_high(out["high"], 20)
    out["lo_20"] = rolling_low(out["low"], 20)
    out["hi_52w"] = rolling_high(out["high"], 252)
    out["lo_52w"] = rolling_low(out["low"], 252)
    return out
