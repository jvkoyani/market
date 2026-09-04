import numpy as np
import pandas as pd
import pytest

from stockdash import indicators, signals, risk


def _trending_df(direction="up", n=300, seed=1):
    rng = np.random.default_rng(seed)
    drift = {"up": 0.002, "down": -0.002, "flat": 0.0}[direction]
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.008, n))))
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.008, "low": close * 0.992, "close": close,
        "volume": rng.integers(5000, 20000, n),
    })
    df.index = pd.bdate_range("2023-01-01", periods=n)
    return indicators.compute_all(df)


def test_uptrend_generates_buy():
    sig = signals.generate("TEST", _trending_df("up"))
    assert sig is not None
    assert sig.action == "BUY"
    assert sig.trend == "UP"
    assert sig.target > sig.price > sig.stop_loss
    assert sig.risk_reward >= 2.0


def test_downtrend_not_buy():
    sig = signals.generate("TEST", _trending_df("down"))
    assert sig is not None
    assert sig.action in ("SELL", "HOLD")
    assert sig.action != "BUY"


def test_signal_none_on_short_history():
    df = _trending_df("up", n=10)
    assert signals.generate("TEST", df) is None


def test_position_size_respects_risk_budget():
    pos = risk.position_size(
        capital=1_000_000, entry=100, stop_loss=95, target=115,
        risk_per_trade=0.01,
    )
    assert pos is not None
    # Risk should not exceed ~1% of capital.
    assert pos.risk_amount <= 1_000_000 * 0.01 + 1e-6
    assert pos.shares > 0


def test_position_size_invalid_stop():
    # stop above entry -> no valid risk
    assert risk.position_size(1_000_000, 100, 105, 120) is None


def test_position_capped_by_max_position_pct():
    pos = risk.position_size(
        capital=1_000_000, entry=100, stop_loss=99.9, target=105,
        risk_per_trade=0.10, max_position_pct=0.15,
    )
    assert pos is not None
    # Tiny stop would blow past sizing on risk alone; cap holds it to 15%.
    assert pos.capital_used <= 1_000_000 * 0.15 + 100


def test_regime_filter_risk_off_reduces_exposure():
    # Build a clearly declining index.
    n = 300
    close = pd.Series(np.linspace(200, 100, n))
    idx = pd.DataFrame({"open": close, "high": close, "low": close,
                        "close": close, "volume": 1})
    idx.index = pd.bdate_range("2023-01-01", periods=n)
    label, mult = risk.market_regime(idx)
    assert label == "RISK_OFF"
    assert mult < 0.5


def test_build_portfolio_caps_total_exposure():
    sigs = [signals.Signal("S%d" % i, "BUY", 100, 115, 95, 3.0, 80, 60, 25,
                           "UP", "x", 2.0) for i in range(50)]
    port = risk.build_portfolio(1_000_000, sigs, index_df=None,
                                max_exposure=0.8, max_positions=12)
    assert port["num_positions"] <= 12
    assert port["deployed"] <= 1_000_000 * 0.8 + 1
    assert port["cash"] >= 0
