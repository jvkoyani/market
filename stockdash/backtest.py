"""Lightweight event-driven backtest.

Walks each symbol's history bar by bar, entering when the signal engine says
``BUY`` and exiting on target, stop, or a max holding period. It applies the
same fixed-fractional sizing as the live engine so the equity curve reflects
the real risk rules -- this is how we sanity-check that the system has an edge
before trusting it with capital.

Not a substitute for a full portfolio simulator, but enough to see whether the
signal + risk logic is net-positive and how deep the drawdowns get.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators, signals
from .risk import position_size


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry: float
    exit: float
    shares: int
    pnl: float
    r_multiple: float      # pnl in units of initial risk
    reason: str            # TARGET | STOP | TIME


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    start_capital: float = 0.0
    end_capital: float = 0.0

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return 100 * wins / len(self.trades)

    @property
    def total_return_pct(self) -> float:
        if not self.start_capital:
            return 0.0
        return 100 * (self.end_capital - self.start_capital) / self.start_capital

    @property
    def avg_r(self) -> float:
        if not self.trades:
            return 0.0
        return float(np.mean([t.r_multiple for t in self.trades]))

    @property
    def profit_factor(self) -> float:
        gains = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = -sum(t.pnl for t in self.trades if t.pnl < 0)
        return gains / losses if losses > 0 else float("inf")

    def summary(self) -> dict:
        return {
            "trades": self.n,
            "win_rate_pct": round(self.win_rate, 1),
            "total_return_pct": round(self.total_return_pct, 2),
            "avg_r_multiple": round(self.avg_r, 2),
            "profit_factor": round(self.profit_factor, 2)
            if self.profit_factor != float("inf") else None,
            "start_capital": round(self.start_capital, 2),
            "end_capital": round(self.end_capital, 2),
        }


def backtest_symbol(symbol: str, df: pd.DataFrame, capital: float = 100_000,
                    max_hold: int = 40, risk_per_trade: float = 0.01,
                    indicator_params: dict | None = None,
                    signal_cfg: dict | None = None,
                    warmup: int = 200) -> list[Trade]:
    """Run the strategy over one symbol's history. Returns closed trades.

    Only one open position per symbol at a time (no pyramiding). Capital is
    passed in only to size positions consistently; PnL is reported per trade.
    """
    enriched = indicators.compute_all(df, indicator_params)
    trades: list[Trade] = []
    in_pos = False
    entry_price = stop = target = 0.0
    shares = 0
    entry_date = None
    bars_held = 0

    idx = enriched.index
    for i in range(warmup, len(enriched)):
        window = enriched.iloc[: i + 1]
        row = enriched.iloc[i]
        price = float(row.close)

        if in_pos:
            bars_held += 1
            hit_stop = row.low <= stop
            hit_target = row.high >= target
            reason = None
            exit_price = price
            # Conservative: if both hit in the same bar, assume stop first.
            if hit_stop:
                reason, exit_price = "STOP", stop
            elif hit_target:
                reason, exit_price = "TARGET", target
            elif bars_held >= max_hold:
                reason, exit_price = "TIME", price
            if reason:
                pnl = shares * (exit_price - entry_price)
                risk = shares * (entry_price - stop)
                r_mult = pnl / risk if risk > 0 else 0.0
                trades.append(Trade(
                    symbol=symbol, entry_date=entry_date, exit_date=idx[i],
                    entry=round(entry_price, 2), exit=round(exit_price, 2),
                    shares=shares, pnl=round(pnl, 2), r_multiple=round(r_mult, 2),
                    reason=reason,
                ))
                in_pos = False
            continue

        sig = signals.generate(symbol, window, signal_cfg)
        if sig and sig.action == "BUY":
            pos = position_size(capital, sig.price, sig.stop_loss, sig.target,
                                symbol, risk_per_trade=risk_per_trade)
            if pos and pos.shares > 0:
                in_pos = True
                entry_price, stop, target = sig.price, sig.stop_loss, sig.target
                shares = pos.shares
                entry_date = idx[i]
                bars_held = 0
    return trades


def run(symbols_data: dict[str, pd.DataFrame], capital: float = 100_000,
        **kwargs) -> BacktestResult:
    """Backtest across several symbols and aggregate.

    ``symbols_data`` maps symbol -> OHLCV frame. Capital is applied per symbol
    for sizing; the aggregate end_capital sums per-trade PnL onto the start.
    """
    all_trades: list[Trade] = []
    for sym, df in symbols_data.items():
        if df is None or len(df) < 260:
            continue
        all_trades.extend(backtest_symbol(sym, df, capital=capital, **kwargs))
    all_trades.sort(key=lambda t: t.exit_date)
    total_pnl = sum(t.pnl for t in all_trades)
    return BacktestResult(
        trades=all_trades,
        start_capital=capital,
        end_capital=capital + total_pnl,
    )
