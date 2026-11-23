"""Risk management and position sizing.

This is the part of the system that actually protects the quarter. The rules:

  * **Fixed-fractional risk**: never risk more than ``risk_per_trade`` (default
    1%) of total capital on a single position. Position size is derived from the
    distance to the stop, *not* from a fixed rupee amount -- so a wide stop
    automatically gets a smaller position.
  * **Exposure cap**: total deployed capital is limited (default 80%), keeping
    a cash buffer for drawdowns and better setups.
  * **Per-position cap**: no single name exceeds ``max_position_pct`` of the
    book (default 15%), enforcing diversification.
  * **Regime filter**: when the market itself is weak (see
    :func:`market_regime`), max exposure is throttled down toward cash.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators


@dataclass
class SizedPosition:
    symbol: str
    entry: float
    stop_loss: float
    target: float
    shares: int
    capital_used: float
    risk_amount: float          # rupees at risk if stop hit
    risk_pct_of_book: float
    reward_amount: float        # rupees gained if target hit


def position_size(capital: float, entry: float, stop_loss: float, target: float,
                  symbol: str = "", risk_per_trade: float = 0.01,
                  max_position_pct: float = 0.15) -> SizedPosition | None:
    """Size a single long position under fixed-fractional risk.

    Returns ``None`` if the stop is not below entry (no valid risk to size on).
    """
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0 or entry <= 0:
        return None

    risk_budget = capital * risk_per_trade
    shares_by_risk = risk_budget / risk_per_share
    shares_by_cap = (capital * max_position_pct) / entry
    shares = int(max(0, min(shares_by_risk, shares_by_cap)))
    if shares <= 0:
        return None

    capital_used = shares * entry
    risk_amount = shares * risk_per_share
    reward_amount = shares * (target - entry)
    return SizedPosition(
        symbol=symbol,
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        target=round(target, 2),
        shares=shares,
        capital_used=round(capital_used, 2),
        risk_amount=round(risk_amount, 2),
        risk_pct_of_book=round(100 * risk_amount / capital, 2),
        reward_amount=round(reward_amount, 2),
    )


def market_regime(index_df: pd.DataFrame | None) -> tuple[str, float]:
    """Classify the broad market from an index (e.g. NIFTY) OHLCV frame.

    Returns ``(label, exposure_multiplier)`` where the multiplier scales the
    max exposure. In a downtrend we deliberately hold more cash.

        RISK_ON      -> 1.0   (index above rising 200 EMA)
        NEUTRAL      -> 0.6
        RISK_OFF     -> 0.25  (index below 200 EMA)
    """
    if index_df is None or len(index_df) < 200:
        return "UNKNOWN", 0.6
    close = index_df["close"]
    ema200 = indicators.ema(close, 200)
    ema50 = indicators.ema(close, 50)
    last = close.iloc[-1]
    e200, e50 = ema200.iloc[-1], ema50.iloc[-1]
    rising = ema200.iloc[-1] > ema200.iloc[-21]  # ~1 month slope

    if last > e200 and e50 > e200 and rising:
        return "RISK_ON", 1.0
    if last < e200 and e50 < e200:
        return "RISK_OFF", 0.25
    return "NEUTRAL", 0.6


def build_portfolio(capital: float, buy_signals: list, index_df=None,
                    risk_per_trade: float = 0.01, max_position_pct: float = 0.15,
                    max_exposure: float = 0.80, max_positions: int = 12) -> dict:
    """Turn a ranked list of BUY signals into a sized, risk-capped portfolio.

    ``buy_signals`` is an iterable of :class:`signals.Signal` (already sorted by
    conviction, best first). Applies the regime filter and all exposure caps.
    Returns a dict with the positions and book-level stats.
    """
    regime, mult = market_regime(index_df)
    exposure_cap = capital * max_exposure * mult

    positions: list[SizedPosition] = []
    deployed = 0.0
    total_risk = 0.0
    for sig in buy_signals:
        if len(positions) >= max_positions:
            break
        pos = position_size(
            capital, sig.price, sig.stop_loss, sig.target, sig.symbol,
            risk_per_trade=risk_per_trade, max_position_pct=max_position_pct,
        )
        if pos is None:
            continue
        if deployed + pos.capital_used > exposure_cap:
            continue
        positions.append(pos)
        deployed += pos.capital_used
        total_risk += pos.risk_amount

    return {
        "regime": regime,
        "exposure_multiplier": mult,
        "capital": round(capital, 2),
        "deployed": round(deployed, 2),
        "cash": round(capital - deployed, 2),
        "deployed_pct": round(100 * deployed / capital, 2) if capital else 0.0,
        "total_risk": round(total_risk, 2),
        "total_risk_pct": round(100 * total_risk / capital, 2) if capital else 0.0,
        "num_positions": len(positions),
        "positions": positions,
    }
