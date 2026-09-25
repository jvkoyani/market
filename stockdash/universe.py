"""NSE stock universe.

Yahoo Finance tickers for NSE stocks use the ``.NS`` suffix (e.g.
``RELIANCE.NS``). We ship a static Nifty-50 list (stable, no network needed)
and can optionally pull the wider Nifty-500 constituents live from NSE when
the user runs with network access.

Keeping a static core list means the dashboard always has *something* to work
with even if the live constituent download is blocked.
"""
from __future__ import annotations

import io

# Nifty 50 core (symbols without the .NS suffix). Refreshed periodically.
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
    "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO", "NESTLEIND",
    "ONGC", "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "GRASIM", "HDFCLIFE", "SBILIFE",
    "BRITANNIA", "DRREDDY", "CIPLA", "DIVISLAB", "EICHERMOT", "HEROMOTOCO",
    "HINDALCO", "INDUSINDBK", "BPCL", "TATACONSUM", "APOLLOHOSP", "BAJAJ-AUTO",
    "TECHM", "LTIM", "M&M", "SHRIRAMFIN",
]

# A broader hand-maintained large/mid-cap set for those who want more breadth
# without a live constituent download. Extends (does not replace) Nifty 50.
NIFTY_NEXT_EXTRA = [
    "DMART", "PIDILITIND", "GODREJCP", "DABUR", "MARICO", "COLPAL", "BERGEPAINT",
    "SIEMENS", "HAVELLS", "BOSCHLTD", "AMBUJACEM", "SHREECEM", "ACC", "DLF",
    "GAIL", "IOC", "VEDL", "ADANIGREEN", "ADANIPOWER", "PNB", "BANKBARODA",
    "IDFCFIRSTB", "CANBK", "AUBANK", "BANDHANBNK", "FEDERALBNK", "CHOLAFIN",
    "MUTHOOTFIN", "ICICIGI", "ICICIPRULI", "SBICARD", "NAUKRI", "ZOMATO",
    "PAYTM", "NYKAA", "IRCTC", "INDIGO", "TRENT", "PAGEIND", "PIIND",
    "DEEPAKNTR", "SRF", "UPL", "AARTIIND", "TATAPOWER", "TORNTPHARM",
    "LUPIN", "AUROPHARMA", "BIOCON", "ALKEM", "MPHASIS", "PERSISTENT",
    "COFORGE", "OFSS", "TVSMOTOR", "BALKRISIND", "MRF", "ASHOKLEY",
    "MOTHERSON", "BHARATFORG", "CUMMINSIND", "ABB", "POLYCAB", "APLAPOLLO",
    "JINDALSTEL", "SAIL", "NMDC", "HINDPETRO", "PETRONET", "IGL", "MGL",
]


def to_yahoo(symbols: list[str]) -> list[str]:
    """Map bare NSE symbols to Yahoo Finance ``.NS`` tickers."""
    return [f"{s}.NS" for s in symbols]


def get_universe(name: str = "nifty50") -> list[str]:
    """Return a list of Yahoo tickers for a named universe.

    name: 'nifty50' | 'nifty100' (approx) | 'broad'
    """
    name = name.lower()
    if name in ("nifty50", "n50"):
        syms = NIFTY_50
    elif name in ("nifty100", "n100", "broad"):
        # de-dup while preserving order
        seen, syms = set(), []
        for s in NIFTY_50 + NIFTY_NEXT_EXTRA:
            if s not in seen:
                seen.add(s)
                syms.append(s)
    else:
        syms = NIFTY_50
    return to_yahoo(syms)


def fetch_nifty500_live(timeout: int = 15) -> list[str]:
    """Best-effort live download of Nifty-500 constituents from NSE.

    Requires network access to nsindia archives. Returns Yahoo tickers, or an
    empty list on any failure (caller should fall back to :func:`get_universe`).
    """
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        import requests  # imported lazily so offline mode has no hard dep

        r = requests.get(
            url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        import csv

        rows = list(csv.DictReader(io.StringIO(r.text)))
        syms = [row["Symbol"].strip() for row in rows if row.get("Symbol")]
        return to_yahoo(syms)
    except Exception:
        return []
