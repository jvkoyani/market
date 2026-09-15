import numpy as np
import pandas as pd
import pytest

from stockdash import indicators as ind


@pytest.fixture
def ohlc():
    n = 300
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n))))
    high = close * 1.01
    low = close * 0.99
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": high, "low": low, "close": close,
        "volume": rng.integers(1000, 5000, n),
    })
    df.index = pd.bdate_range("2023-01-01", periods=n)
    return df


def test_ema_matches_pandas(ohlc):
    e = ind.ema(ohlc["close"], 20)
    expected = ohlc["close"].ewm(span=20, adjust=False).mean()
    assert np.allclose(e, expected)


def test_rsi_bounds(ohlc):
    r = ind.rsi(ohlc["close"]).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1, 60, dtype=float))  # strictly increasing
    r = ind.rsi(close).dropna()
    assert (r > 99).all()


def test_rsi_all_losses_is_low():
    close = pd.Series(np.arange(60, 1, -1, dtype=float))  # strictly decreasing
    r = ind.rsi(close).dropna()
    assert (r < 1).all()


def test_atr_positive(ohlc):
    a = ind.atr(ohlc).dropna()
    assert (a > 0).all()


def test_macd_hist_is_line_minus_signal(ohlc):
    line, sig, hist = ind.macd(ohlc["close"])
    assert np.allclose((line - sig).dropna(), hist.dropna())


def test_adx_bounds(ohlc):
    a = ind.adx(ohlc).dropna()
    assert (a >= 0).all() and (a <= 100).all()


def test_compute_all_has_columns(ohlc):
    out = ind.compute_all(ohlc)
    for col in ["ema20", "ema50", "ema200", "rsi", "macd_hist", "atr", "adx",
                "bb_upper", "bb_lower", "roc", "vol_sma", "hi_20", "hi_52w"]:
        assert col in out.columns
    assert len(out) == len(ohlc)
