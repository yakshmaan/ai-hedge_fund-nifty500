"""
backtester_v5.py
----------------
Backtester V5 — 100% Passive Buy & Hold (Top 15 Nifty stocks by volume).
 
Strategy:
  - On day 1, buy top 15 Nifty 50 stocks equally with all capital
  - Never trade again — pure buy and hold
  - Compare against V4 active system on return, alpha, and max drawdown
 
Usage:
    python backtester_v5.py
"""
 
import sqlite3
import pandas as pd
import numpy as np
import sys
import os
 
DB_PATH         = "data/nifty500.db"
INITIAL_CAPITAL = 100_000.0
BACKTEST_START  = "2018-01-01"
BACKTEST_END    = "2026-04-16"
COMMISSION_PCT  = 0.001
INDEX_TOP_N     = 48
 
 
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"""
        SELECT Date, Symbol, Close, Volume, Daily_Return
        FROM prices WHERE Date BETWEEN '{BACKTEST_START}' AND '{BACKTEST_END}'
        ORDER BY Symbol, Date
    """, conn)
    conn.close()
    return df
 
 
def get_top_n_symbols(df, n=INDEX_TOP_N):
    """Top N stocks by average volume over full period."""
    avg_vol = df.groupby("Symbol")["Volume"].mean().nlargest(n)
    return list(avg_vol.index)
 
 
def run_backtest_v5():
    print(f"\nV5 Passive Backtest ({BACKTEST_START} to {BACKTEST_END})...")
    df        = load_data()
    all_dates = sorted(df["Date"].unique())
    top_syms  = get_top_n_symbols(df, INDEX_TOP_N)
 
    print(f"  Holding top {INDEX_TOP_N} stocks: {', '.join(top_syms)}\n")
 
    # Buy equally on day 1
    first_day    = all_dates[0]
    first_prices = df[df["Date"] == first_day].set_index("Symbol")["Close"].to_dict()
    per_stock    = INITIAL_CAPITAL / INDEX_TOP_N
    holdings     = {}  # symbol -> shares
 
    cash = INITIAL_CAPITAL
    for sym in top_syms:
        price = first_prices.get(sym)
        if price and price > 0:
            shares = int(per_stock / (price * (1 + COMMISSION_PCT)))
            cost   = shares * price * (1 + COMMISSION_PCT)
            if shares > 0 and cost <= cash:
                holdings[sym] = {"shares": shares, "entry_price": price}
                cash -= cost
 
    # Track equity curve day by day
    equity_curve = []
    peak_value   = INITIAL_CAPITAL
 
    for date in all_dates:
        day_prices = df[df["Date"] == date].set_index("Symbol")["Close"].to_dict()
        port_val   = cash + sum(
            holdings[s]["shares"] * day_prices.get(s, holdings[s]["entry_price"])
            for s in holdings
        )
        if port_val > peak_value:
            peak_value = port_val
        drawdown = (peak_value - port_val) / peak_value
 
        equity_curve.append({
            "date":            date,
            "portfolio_value": port_val,
            "drawdown":        -drawdown,
        })
 
    return equity_curve, holdings
 
 
def compute_metrics(equity_curve, df):
    equity       = pd.DataFrame(equity_curve).set_index("date")
    equity.index = pd.to_datetime(equity.index)
    final_val    = equity["portfolio_value"].iloc[-1]
    total_ret    = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL
    daily_ret    = equity["portfolio_value"].pct_change().dropna()
    rf_daily     = 0.06 / 252
    excess       = daily_ret - rf_daily
    sharpe       = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
    downside     = excess[excess < 0]
    sortino      = (excess.mean() / downside.std()) * np.sqrt(252) if downside.std() > 0 else 0
    roll_max     = equity["portfolio_value"].cummax()
    max_dd       = float(((equity["portfolio_value"] - roll_max) / roll_max).min())
    years        = (equity.index[-1] - equity.index[0]).days / 365
    annual       = (1 + total_ret) ** (1 / max(years, 1)) - 1
    calmar       = annual / abs(max_dd) if max_dd != 0 else 0
    benchmark    = df.groupby("Date")["Close"].mean()
    bench_ret    = (benchmark.iloc[-1] - benchmark.iloc[0]) / benchmark.iloc[0]
    bench_roll   = benchmark.cummax()
    bench_max_dd = float(((benchmark - bench_roll) / bench_roll).min())
 
    return {
        "initial_capital":  INITIAL_CAPITAL,
        "final_value":      round(final_val, 2),
        "total_return":     round(total_ret * 100, 2),
        "annual_return":    round(annual * 100, 2),
        "benchmark_return": round(bench_ret * 100, 2),
        "alpha":            round((total_ret - bench_ret) * 100, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "sortino_ratio":    round(sortino, 3),
        "calmar_ratio":     round(calmar, 3),
        "max_drawdown":     round(max_dd * 100, 2),
        "bench_max_dd":     round(bench_max_dd * 100, 2),
    }
 
 
def print_report(metrics, equity_curve):
    print("\n" + "═" * 58)
    print("  BACKTESTER V5 — PASSIVE BUY & HOLD REPORT")
    print(f"  {BACKTEST_START} to {BACKTEST_END}  [Top {INDEX_TOP_N} stocks, equal weight]")
    print("=" * 58)
    print(f"  Initial Capital     : ₹{metrics['initial_capital']:>12,.0f}")
    print(f"  Final Value         : ₹{metrics['final_value']:>12,.0f}")
    print(f"  {'─'*42}")
    print(f"  Total Return        : {metrics['total_return']:>+11.2f}%")
    print(f"  Annual Return       : {metrics['annual_return']:>+11.2f}%")
    print(f"  Benchmark Return    : {metrics['benchmark_return']:>+11.2f}%")
    print(f"  Alpha vs Benchmark  : {metrics['alpha']:>+11.2f}%")
    print(f"  {'─'*42}")
    print(f"  Sharpe Ratio        : {metrics['sharpe_ratio']:>12.3f}")
    print(f"  Sortino Ratio       : {metrics['sortino_ratio']:>12.3f}")
    print(f"  Calmar Ratio        : {metrics['calmar_ratio']:>12.3f}")
    print(f"  {'─'*42}")
    print(f"  V5 Max Drawdown     : {metrics['max_drawdown']:>+11.2f}%")
    print(f"  Benchmark Max DD    : {metrics['bench_max_dd']:>+11.2f}%")
    print("=" * 58)
    print("\n  ── V4 vs V5 vs Benchmark ──────────────────────────")
    print(f"  {'Metric':<22} {'V4':>10} {'V5 Passive':>12} {'Benchmark':>12}")
    print(f"  {'─'*58}")
    print(f"  {'Total Return':<22} {'  +66.34%':>10} {metrics['total_return']:>+11.2f}% {'  +176.48%':>12}")
    print(f"  {'Max Drawdown':<22} {'  -16.19%':>10} {metrics['max_drawdown']:>+11.2f}% {metrics['bench_max_dd']:>+11.2f}%")
    print("=" * 58)
 
    pd.DataFrame(equity_curve).to_csv("data/equity_curve_v5.csv", index=False)
    print(f"\n  Equity curve → data/equity_curve_v5.csv")
 
 
if __name__ == "__main__":
    print("Full Nifty 50 V5 — Passive Buy & Hold backtest...")
    equity_curve, holdings = run_backtest_v5()
    df      = load_data()
    metrics = compute_metrics(equity_curve, df)
    print_report(metrics, equity_curve)
 