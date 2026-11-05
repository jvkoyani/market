"""Indian Stock Market Dashboard -- Streamlit front-end.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Two data modes (sidebar):
  * Live (yfinance)  -- real NSE data, needs internet.
  * Offline sample   -- bundled synthetic data, works anywhere (demo only).

The dashboard screens the chosen universe, shows BUY/SELL/HOLD calls with
entry / target / stop-loss, sizes a risk-managed portfolio, lets you drill
into any stock's chart, and can backtest the strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockdash import data as data_mod
from stockdash import indicators, screener, risk, backtest, universe, signals

st.set_page_config(page_title="India Stock Signals", page_icon="📈", layout="wide")

# ---- palette (readable in Streamlit's default light theme) --------------
GREEN, RED, AMBER, INK, MUTED = "#0F9D58", "#D93025", "#F4B400", "#202124", "#5F6368"
ACTION_COLOR = {"BUY": GREEN, "SELL": RED, "HOLD": AMBER}


# ---- data plumbing -------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_screen(symbols, source, period):
    prog = st.progress(0.0, text="Analysing universe…")

    def _p(done, total, sym):
        prog.progress(done / total, text=f"Analysing {sym} ({done}/{total})")

    df = screener.screen(list(symbols), source=source, period=period, progress=_p)
    prog.empty()
    return df


@st.cache_data(show_spinner=False)
def load_one(symbol, source, period):
    return data_mod.get_history(symbol, source=source, period=period)


def load_index(source):
    if source == "sample":
        return data_mod.load_sample("_NSEI")
    return data_mod.get_history("^NSEI", source="yfinance", period="2y")


def resolve_universe(choice, source):
    if source == "sample":
        syms = [s for s in data_mod.available_sample_symbols() if not s.startswith("_")]
        return syms
    if choice == "Nifty 500 (live)":
        live = universe.fetch_nifty500_live()
        if live:
            return live
        st.warning("Couldn't fetch Nifty 500 list; falling back to broad set.")
        return universe.get_universe("broad")
    if choice == "Nifty 100 (broad)":
        return universe.get_universe("broad")
    return universe.get_universe("nifty50")


# ---- sidebar -------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
source_label = st.sidebar.radio(
    "Data source", ["Offline sample (demo)", "Live NSE (yfinance)"], index=0,
    help="Live mode needs internet and `pip install yfinance`.",
)
source = "sample" if source_label.startswith("Offline") else "yfinance"

uni_choice = st.sidebar.selectbox(
    "Universe", ["Nifty 50", "Nifty 100 (broad)", "Nifty 500 (live)"],
    disabled=(source == "sample"),
    help="Offline mode uses whatever is in sample_data/.",
)
period = st.sidebar.selectbox("History window", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("### 💰 Portfolio & risk")
capital = st.sidebar.number_input("Capital (₹)", 10_000, 100_000_000, 10_00_000,
                                  step=50_000, format="%d")
risk_per_trade = st.sidebar.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25) / 100
max_pos_pct = st.sidebar.slider("Max per position (%)", 5, 30, 15) / 100
max_exposure = st.sidebar.slider("Max total exposure (%)", 20, 100, 80) / 100
max_positions = st.sidebar.slider("Max # positions", 3, 25, 12)

st.sidebar.markdown("### 🎯 Signal tuning")
buy_score = st.sidebar.slider("BUY score threshold", 40, 90, 60)
target_mult = st.sidebar.slider("Target = entry + N×ATR", 1.5, 5.0, 3.0, 0.5)
stop_mult = st.sidebar.slider("Stop = entry − N×ATR", 0.5, 3.0, 1.5, 0.25)

signal_cfg = {"buy_score": buy_score, "target_atr_mult": target_mult,
              "stop_atr_mult": stop_mult}

# ---- header --------------------------------------------------------------
st.title("📈 Indian Stock Market — Signal Dashboard")
st.caption(
    "Multi-factor technical analysis → BUY / SELL / HOLD with entry, target & "
    "stop-loss, plus risk-managed position sizing. "
    "**Educational tool, not investment advice.**"
)

symbols = resolve_universe(uni_choice, source)
if not symbols:
    st.error("No symbols to analyse. In offline mode, run `python generate_sample_data.py`.")
    st.stop()

with st.spinner("Screening…"):
    screen_df = run_screen(tuple(symbols), source, period)

if screen_df.empty:
    st.error("No data returned. If live, check your connection / ticker suffixes.")
    st.stop()

# ---- market regime banner -----------------------------------------------
index_df = load_index(source)
regime, mult = risk.market_regime(index_df)
regime_msg = {
    "RISK_ON": ("🟢 RISK-ON", "Market above rising 200-EMA — full exposure allowed."),
    "NEUTRAL": ("🟡 NEUTRAL", "Mixed market — exposure throttled to ~60%."),
    "RISK_OFF": ("🔴 RISK-OFF", "Market below 200-EMA — mostly cash (~25% max)."),
    "UNKNOWN": ("⚪ UNKNOWN", "Not enough index history to judge regime."),
}[regime]
st.info(f"**Market regime: {regime_msg[0]}** — {regime_msg[1]}")

# ---- KPI row -------------------------------------------------------------
buys = screen_df[screen_df.action == "BUY"]
sells = screen_df[screen_df.action == "SELL"]
holds = screen_df[screen_df.action == "HOLD"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Stocks analysed", len(screen_df))
k2.metric("🟢 BUY signals", len(buys))
k3.metric("🔴 SELL / avoid", len(sells))
k4.metric("🟡 HOLD", len(holds))

tab_signals, tab_portfolio, tab_detail, tab_backtest = st.tabs(
    ["🔎 Signals", "💼 Suggested Portfolio", "🔬 Stock Detail", "🧪 Backtest"]
)

# ================= SIGNALS TAB ===========================================
with tab_signals:
    st.subheader("Ranked signals")
    only = st.multiselect("Show actions", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
    view = screen_df[screen_df.action.isin(only)].copy()
    view["symbol"] = view["symbol"].str.replace(".NS", "", regex=False)
    show_cols = ["symbol", "action", "price", "target", "stop_loss", "risk_reward",
                 "score", "trend", "rsi", "adx", "atr_pct", "reasons"]

    def _style(row):
        return [f"color: {ACTION_COLOR.get(row.action, INK)}; font-weight: 700"
                if c == "action" else "" for c in show_cols]

    st.dataframe(
        view[show_cols].style.apply(_style, axis=1).format({
            "price": "₹{:.2f}", "target": "₹{:.2f}", "stop_loss": "₹{:.2f}",
            "risk_reward": "{:.1f}", "score": "{:.0f}", "rsi": "{:.0f}",
            "adx": "{:.0f}", "atr_pct": "{:.1f}%",
        }),
        width='stretch', height=520,
    )
    st.download_button("⬇️ Download signals (CSV)", view[show_cols].to_csv(index=False),
                       "signals.csv", "text/csv")

# ================= PORTFOLIO TAB =========================================
with tab_portfolio:
    st.subheader("Risk-managed portfolio from today's BUYs")
    buy_sigs = screener.top_buys(screen_df)
    port = risk.build_portfolio(
        capital, buy_sigs, index_df=index_df, risk_per_trade=risk_per_trade,
        max_position_pct=max_pos_pct, max_exposure=max_exposure,
        max_positions=max_positions,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions", port["num_positions"])
    c2.metric("Capital deployed", f"₹{port['deployed']:,.0f}", f"{port['deployed_pct']:.0f}%")
    c3.metric("Cash buffer", f"₹{port['cash']:,.0f}")
    c4.metric("Total risk if all stops hit", f"₹{port['total_risk']:,.0f}",
              f"{port['total_risk_pct']:.2f}% of capital", delta_color="inverse")

    if port["positions"]:
        pos_df = pd.DataFrame([p.__dict__ for p in port["positions"]])
        pos_df["symbol"] = pos_df["symbol"].str.replace(".NS", "", regex=False)
        pos_df["potential_reward_%"] = 100 * pos_df["reward_amount"] / capital
        st.dataframe(
            pos_df.rename(columns={
                "entry": "entry ₹", "stop_loss": "stop ₹", "target": "target ₹",
                "capital_used": "capital ₹", "risk_amount": "risk ₹",
                "reward_amount": "reward ₹", "risk_pct_of_book": "risk %",
            })[["symbol", "shares", "entry ₹", "stop ₹", "target ₹",
                "capital ₹", "risk ₹", "risk %", "reward ₹"]].style.format({
                "entry ₹": "{:.2f}", "stop ₹": "{:.2f}", "target ₹": "{:.2f}",
                "capital ₹": "{:,.0f}", "risk ₹": "{:,.0f}", "risk %": "{:.2f}",
                "reward ₹": "{:,.0f}",
            }),
            width='stretch',
        )
        st.caption(
            f"Regime multiplier **{port['exposure_multiplier']:.2f}×** applied to "
            f"max exposure. Worst-case portfolio loss (all stops hit) is capped at "
            f"**{port['total_risk_pct']:.2f}%** of capital — the core defence for a "
            f"positive quarter."
        )
    else:
        st.warning("No positions sized — either no BUY signals or regime forced cash.")

# ================= DETAIL TAB ============================================
with tab_detail:
    st.subheader("Drill into a stock")
    pick = st.selectbox("Symbol", screen_df["symbol"].tolist())
    hist = load_one(pick, source, period)
    if hist is None or hist.empty:
        st.error("No history for this symbol.")
    else:
        enr = indicators.compute_all(hist)
        sig = signals.generate(pick, enr, signal_cfg)
        if sig:
            a, b, c, d, e = st.columns(5)
            a.markdown(f"### :{'green' if sig.action=='BUY' else 'red' if sig.action=='SELL' else 'orange'}[{sig.action}]")
            b.metric("Price", f"₹{sig.price:,.2f}")
            c.metric("Target", f"₹{sig.target:,.2f}")
            d.metric("Stop-loss", f"₹{sig.stop_loss:,.2f}")
            e.metric("Reward:Risk", f"{sig.risk_reward:.1f} : 1")
            st.write(f"**Why:** {sig.reasons}")

        plot = enr.tail(180)
        price_chart = plot[["close", "ema20", "ema50", "ema200"]].rename(columns={
            "close": "Close", "ema20": "EMA20", "ema50": "EMA50", "ema200": "EMA200"})
        st.line_chart(price_chart, height=320)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("RSI (14)")
            st.line_chart(plot[["rsi"]].rename(columns={"rsi": "RSI"}), height=180)
        with cc2:
            st.caption("MACD histogram")
            st.bar_chart(plot[["macd_hist"]].rename(columns={"macd_hist": "MACD hist"}), height=180)

# ================= BACKTEST TAB ==========================================
with tab_backtest:
    st.subheader("Strategy backtest")
    st.caption("Replays the exact signal + risk rules bar-by-bar over history. "
               "In offline mode this uses synthetic data, so it validates the "
               "*mechanics*, not real-world edge.")
    n_bt = st.slider("Symbols to backtest (from top of list)", 5, min(50, len(symbols)),
                     min(20, len(symbols)))
    if st.button("Run backtest"):
        with st.spinner("Backtesting…"):
            sd = {}
            for s in symbols[:n_bt]:
                h = load_one(s, source, period)
                if h is not None and len(h) >= 260:
                    sd[s] = h
            res = backtest.run(sd, capital=capital, risk_per_trade=risk_per_trade,
                               signal_cfg=signal_cfg)
        s = res.summary()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", s["trades"])
        m2.metric("Win rate", f"{s['win_rate_pct']:.0f}%")
        m3.metric("Total return", f"{s['total_return_pct']:.1f}%")
        m4.metric("Profit factor", s["profit_factor"] or "∞")
        if res.trades:
            eq = pd.DataFrame({
                "pnl": [t.pnl for t in res.trades],
            }, index=[t.exit_date for t in res.trades])
            eq["equity"] = capital + eq["pnl"].cumsum()
            st.line_chart(eq[["equity"]], height=280)
            st.dataframe(pd.DataFrame([t.__dict__ for t in res.trades[-25:]]),
                         width='stretch')

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This dashboard is an educational, algorithmic analysis "
    "tool. Signals are generated from historical price patterns and carry no "
    "guarantee of future returns. Markets involve risk of loss. Do your own "
    "research and consult a SEBI-registered advisor before investing."
)
