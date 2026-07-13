"""Build the static dashboard payload for Vercel.

Runs the screener over the chosen universe and writes ``web/signals.json`` --
a single file the static front-end (``web/index.html``) fetches and renders.
Because the output is static JSON + HTML, it deploys to Vercel's CDN with no
server, no WebSockets and no serverless timeouts.

Usage:
    python build_web.py --source sample                 # offline demo data
    python build_web.py --source yfinance --universe nifty50
    python build_web.py --source yfinance --universe broad --period 2y

In CI (GitHub Action) this runs with --source yfinance to refresh real data
on a schedule; locally/offline it falls back to the bundled sample data.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from stockdash import data as data_mod
from stockdash import indicators, screener, risk, universe

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

REGIME_NOTE = {
    "RISK_ON": "Market above rising 200-EMA — full exposure allowed.",
    "NEUTRAL": "Mixed market — exposure throttled to ~60%.",
    "RISK_OFF": "Market below 200-EMA — mostly cash (~25% max).",
    "UNKNOWN": "Not enough index history to judge regime.",
}


def resolve_universe(name: str, source: str) -> tuple[list[str], str]:
    if source == "sample":
        syms = [s for s in data_mod.available_sample_symbols() if not s.startswith("_")]
        return syms, "Sample (offline)"
    if name == "nifty500":
        live = universe.fetch_nifty500_live()
        if live:
            return live, "Nifty 500 (live)"
        return universe.get_universe("broad"), "Nifty 100 (broad, fallback)"
    if name == "broad":
        return universe.get_universe("broad"), "Nifty 100 (broad)"
    return universe.get_universe("nifty50"), "Nifty 50"


def load_index(source: str):
    if source == "sample":
        return data_mod.load_sample("_NSEI")
    return data_mod.get_history("^NSEI", source="yfinance", period="2y")


def build(source: str, uni: str, period: str, spark_len: int = 90) -> dict:
    symbols, uni_label = resolve_universe(uni, source)
    signals_out = []
    for sym in symbols:
        hist = data_mod.get_history_cached(sym, source=source, period=period)
        if hist is None or len(hist) < 60:
            continue
        enriched = indicators.compute_all(hist)
        from stockdash import signals as sig_mod
        sig = sig_mod.generate(sym, enriched)
        if sig is None:
            continue
        d = sig.as_dict()
        d["symbol"] = sym.replace(".NS", "")
        d["ticker"] = sym
        # Compact close-price series for the sparkline / detail chart.
        closes = hist["close"].tail(spark_len).round(2).tolist()
        d["closes"] = closes
        signals_out.append(d)

    # Rank: BUY first (score desc), then HOLD, then SELL.
    rank = {"BUY": 0, "HOLD": 1, "SELL": 2}
    signals_out.sort(key=lambda s: (rank.get(s["action"], 3), -s["score"]))

    index_df = load_index(source)
    regime, mult = risk.market_regime(index_df)

    n_buy = sum(1 for s in signals_out if s["action"] == "BUY")
    n_sell = sum(1 for s in signals_out if s["action"] == "SELL")
    n_hold = sum(1 for s in signals_out if s["action"] == "HOLD")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "universe": uni_label,
        "period": period,
        "regime": {"label": regime, "multiplier": mult, "note": REGIME_NOTE[regime]},
        "kpis": {"analysed": len(signals_out), "buy": n_buy, "sell": n_sell, "hold": n_hold},
        "defaults": {
            "capital": 1_000_000, "risk_per_trade": 0.01, "max_position_pct": 0.15,
            "max_exposure": 0.80, "max_positions": 12,
        },
        "signals": signals_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sample", "yfinance"], default="sample")
    ap.add_argument("--universe", choices=["nifty50", "broad", "nifty500"], default="nifty50")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--out", default=os.path.join(WEB_DIR, "signals.json"))
    args = ap.parse_args()

    payload = build(args.source, args.universe, args.period)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out}  ({kb:.0f} KB)  "
          f"{payload['kpis']['analysed']} stocks, {payload['kpis']['buy']} BUY, "
          f"regime={payload['regime']['label']}")


if __name__ == "__main__":
    main()
