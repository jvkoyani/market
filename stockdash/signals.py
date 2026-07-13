"""Signal generation.

Turns the indicator set from :mod:`indicators` into an actionable call:
``BUY`` / ``SELL`` / ``HOLD`` with an entry, a target, a stop-loss and a
0-100 conviction score.

The philosophy is capital preservation first (see README):
  * Only act on multi-factor agreement, never a single indicator.
  * Every long has a hard ATR-based stop and a target that is at least
    ``min_rr`` times the risk. Trades that cannot clear that reward:risk bar
    are downgraded to HOLD -- a skipped trade never loses money.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Signal:
    symbol: str
    action: str          # BUY | SELL | HOLD
    price: float         # last close / suggested entry
    target: float
    stop_loss: float
    risk_reward: float
    score: float         # 0-100 conviction
    rsi: float
    adx: float
    trend: str           # UP | DOWN | SIDEWAYS
    reasons: str         # human-readable justification
    atr_pct: float       # ATR as % of price (volatility gauge)

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULTS = {
    "target_atr_mult": 3.0,   # target = entry + 3*ATR
    "stop_atr_mult": 1.5,     # stop   = entry - 1.5*ATR  (=> RR 2:1 baseline)
    "min_rr": 2.0,            # reject setups below this reward:risk
    "buy_score": 60.0,        # score threshold to call a BUY
    "sell_score": 40.0,       # below this => SELL/avoid
    "min_adx": 18.0,          # require a minimum of trend strength for BUY
}


def _trend_label(row) -> str:
    if row.close > row.ema50 and row.ema20 > row.ema50:
        return "UP"
    if row.close < row.ema50 and row.ema20 < row.ema50:
        return "DOWN"
    return "SIDEWAYS"


def _score(row) -> tuple[float, list[str]]:
    """Weighted multi-factor score in [0, 100] with reasons.

    Weights are deliberately spread so no single factor dominates.
    """
    score = 0.0
    reasons: list[str] = []

    # --- Trend structure (35) ---------------------------------------
    if row.close > row.ema200:
        score += 12
        reasons.append("above 200 EMA (long-term uptrend)")
    if row.ema20 > row.ema50:
        score += 12
        reasons.append("20 EMA > 50 EMA (medium-term up)")
    if row.close > row.ema20:
        score += 11
        reasons.append("price above 20 EMA")

    # --- Momentum (30) ----------------------------------------------
    if 50 <= row.rsi <= 70:
        score += 12
        reasons.append(f"RSI healthy ({row.rsi:.0f})")
    elif row.rsi > 70:
        reasons.append(f"RSI overbought ({row.rsi:.0f})")
    elif 40 <= row.rsi < 50:
        score += 5
    if row.macd_hist > 0 and row.macd > row.macd_signal:
        score += 10
        reasons.append("MACD bullish")
    if row.roc > 0:
        score += 8
        reasons.append(f"positive {row.roc:.0f}% momentum")

    # --- Trend strength (15) ----------------------------------------
    if row.adx >= 25:
        score += 15
        reasons.append(f"strong trend (ADX {row.adx:.0f})")
    elif row.adx >= 18:
        score += 8
        reasons.append(f"building trend (ADX {row.adx:.0f})")

    # --- Volume confirmation (10) -----------------------------------
    if not np.isnan(row.vol_sma) and row.vol_sma > 0 and row.volume > row.vol_sma:
        score += 10
        reasons.append("above-average volume")

    # --- Breakout kicker (10) ---------------------------------------
    if not np.isnan(row.hi_20) and row.close >= row.hi_20:
        score += 10
        reasons.append("20-day breakout")

    # --- Penalties ---------------------------------------------------
    if row.rsi > 78:
        score -= 10
        reasons.append("extended -- overbought risk")
    if row.close < row.ema200:
        score -= 8

    return float(np.clip(score, 0, 100)), reasons


def generate(symbol: str, df: pd.DataFrame, cfg: dict | None = None) -> Signal | None:
    """Produce a :class:`Signal` from an indicator-enriched frame.

    ``df`` must already have the columns from :func:`indicators.compute_all`.
    Returns ``None`` if there is not enough history to evaluate.
    """
    c = {**DEFAULTS, **(cfg or {})}
    df = df.dropna(subset=["ema50", "atr", "rsi", "adx"])
    if len(df) < 2:
        return None

    row = df.iloc[-1]
    price = float(row.close)
    atr_val = float(row.atr)
    if atr_val <= 0 or np.isnan(atr_val):
        return None

    score, reasons = _score(row)
    trend = _trend_label(row)
    atr_pct = 100 * atr_val / price

    # Risk-managed target / stop from volatility.
    target = price + c["target_atr_mult"] * atr_val
    stop = price - c["stop_atr_mult"] * atr_val
    risk = price - stop
    reward = target - price
    rr = reward / risk if risk > 0 else 0.0

    # Decide action.
    action = "HOLD"
    if score >= c["buy_score"] and trend in ("UP", "SIDEWAYS") and row.adx >= c["min_adx"]:
        if rr >= c["min_rr"]:
            action = "BUY"
        else:
            reasons.append(f"reward:risk too low ({rr:.1f}) -> hold")
    elif score <= c["sell_score"] or trend == "DOWN":
        action = "SELL"
        # For an exit call, a stop above and target below make more sense.
        target = price - c["target_atr_mult"] * atr_val
        stop = price + c["stop_atr_mult"] * atr_val
        if trend == "DOWN":
            reasons.append("downtrend -- avoid / exit longs")

    return Signal(
        symbol=symbol,
        action=action,
        price=round(price, 2),
        target=round(target, 2),
        stop_loss=round(stop, 2),
        risk_reward=round(rr, 2),
        score=round(score, 1),
        rsi=round(float(row.rsi), 1),
        adx=round(float(row.adx), 1),
        trend=trend,
        reasons="; ".join(reasons) if reasons else "no strong edge",
        atr_pct=round(atr_pct, 2),
    )
