"""Screener -- run the analysis across a whole universe.

Ties data + indicators + signals together and returns a ranked table of
opportunities. This is what the dashboard calls to populate its main view.
"""
from __future__ import annotations

import pandas as pd

from . import data as data_mod
from . import indicators, signals


def analyse_symbol(symbol: str, source: str = "yfinance",
                   indicator_params: dict | None = None,
                   signal_cfg: dict | None = None,
                   period: str = "2y") -> signals.Signal | None:
    """Full pipeline for one symbol: fetch -> indicators -> signal."""
    df = data_mod.get_history_cached(symbol, source=source, period=period)
    if df is None or len(df) < 60:
        return None
    enriched = indicators.compute_all(df, indicator_params)
    return signals.generate(symbol, enriched, signal_cfg)


def screen(symbols: list[str], source: str = "yfinance",
           indicator_params: dict | None = None, signal_cfg: dict | None = None,
           period: str = "2y", progress=None) -> pd.DataFrame:
    """Analyse many symbols and return a DataFrame sorted by conviction.

    ``progress`` is an optional callable ``progress(done, total, symbol)`` used
    by the dashboard to render a progress bar.
    """
    rows = []
    total = len(symbols)
    for i, sym in enumerate(symbols, 1):
        if progress:
            progress(i, total, sym)
        sig = analyse_symbol(sym, source, indicator_params, signal_cfg, period)
        if sig is not None:
            rows.append(sig.as_dict())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Rank: BUYs first (by score desc), then HOLD, then SELL.
    action_rank = {"BUY": 0, "HOLD": 1, "SELL": 2}
    df["_rank"] = df["action"].map(action_rank).fillna(3)
    df = df.sort_values(["_rank", "score"], ascending=[True, False])
    return df.drop(columns="_rank").reset_index(drop=True)


def top_buys(screen_df: pd.DataFrame, n: int | None = None) -> list[signals.Signal]:
    """Extract BUY signals (best first) as Signal objects for sizing."""
    if screen_df.empty:
        return []
    buys = screen_df[screen_df["action"] == "BUY"].copy()
    if n:
        buys = buys.head(n)
    out = []
    for _, r in buys.iterrows():
        out.append(signals.Signal(**{k: r[k] for k in signals.Signal.__annotations__}))
    return out
