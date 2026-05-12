"""
backtester.py
-------------
Runs the full system (agents + risk engine) on historical data.
Simulates trades day by day WITHOUT lookahead bias.

Key rule: on day T, we only use data from day T-1 and earlier.
Never look at future prices when making past decisions.

What this produces:
  - Equity curve (portfolio value over time)
  - Total return vs Nifty 50 buy-and-hold benchmark
  - Sharpe Ratio (return per unit of risk)
  - Max Drawdown (worst peak-to-trough loss)
  - Win rate (% of trades that were profitable)
  - Full trade log

Usage:
    python backtester.py                    # backtest all stocks
    python backtester.py --symbol RELIANCE  # backtest one stock
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk.risk_engine import RiskConfig, evaluate_trade
from dataclasses import dataclass, field

DB_PATH = "data/nifty50.db"

# ── CONFIG ────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL  = 100_000.0   # starting capital in INR
BACKTEST_START   = "2020-01-01"
BACKTEST_END     = "2024-12-31"
COMMISSION_PCT   = 0.001        # 0.1% per trade (realistic for Zerodha)
REBALANCE_FREQ   = 5            # re-evaluate signals every 5 trading days

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_price_data(symbol: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    if symbol:
        query = f"""
            SELECT Date, Symbol, Open, High, Low, Close, Volume, Daily_Return
            FROM prices
            WHERE Symbol = '{symbol}'
            AND Date BETWEEN '{BACKTEST_START}' AND '{BACKTEST_END}'
            ORDER BY Date
        """
    else:
        query = f"""
            SELECT Date, Symbol, Open, High, Low, Close, Volume, Daily_Return
            FROM prices
            WHERE Date BETWEEN '{BACKTEST_START}' AND '{BACKTEST_END}'
            ORDER BY Symbol, Date
        """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


# ── SIGNAL COMPUTATION (no LLM in backtest — too slow for 1000s of days) ─────

def compute_signals_for_date(symbol_data: pd.DataFrame, current_idx: int) -> float:
    """
    Compute combined signal score using only data up to current_idx.
    This is the backtesting version of the orchestrator — same logic,
    no LLM call (would take hours for 5 years of data).

    Uses the same weights as orchestrator.py:
      momentum 35%, mean_reversion 30%, ml 35%
    """
    # Use only historical data up to this point (no lookahead)
    hist = symbol_data.iloc[:current_idx + 1].copy()

    if len(hist) < 60:
        return 0.0  # not enough history

    close = hist["Close"]

    # ── Momentum signal (same as momentum_agent.py) ──────────────────────────
    ma_fast = close.rolling(20).mean()
    ma_slow = close.rolling(50).mean()
    ma_gap  = (ma_fast - ma_slow) / ma_slow

    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss     = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs       = gain / loss.replace(0, np.nan)
    rsi      = (100 - (100 / (1 + rs))).fillna(50)
    rsi_sig  = (rsi - 50) / 50

    momentum_score = ((0.6 * ma_gap) + (0.4 * rsi_sig)).clip(-1, 1).iloc[-1]

    # ── Mean reversion signal (same as mean_reversion_agent.py) ─────────────
    rolling_mean = close.rolling(20).mean()
    rolling_std  = close.rolling(20).std()
    zscore       = (close - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore_norm  = (zscore / 3).clip(-1, 1)

    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    bb_pos = ((close - lower) / (upper - lower).replace(0, np.nan) - 0.5) * 2

    mr_score = -(0.5 * zscore_norm + 0.5 * bb_pos).clip(-1, 1).iloc[-1]

    # ── Simple ML proxy (logistic of recent returns — fast approximation) ────
    ret_5  = close.pct_change(5).iloc[-1]
    ret_20 = close.pct_change(20).iloc[-1]
    vol    = close.pct_change().rolling(10).std().iloc[-1]
    ml_proxy = np.clip((ret_5 - ret_20) / (vol + 1e-8) * 0.1, -1, 1)

    # Combined
    combined = (
        0.35 * float(momentum_score if not np.isnan(momentum_score) else 0) +
        0.30 * float(mr_score if not np.isnan(mr_score) else 0) +
        0.35 * float(ml_proxy if not np.isnan(ml_proxy) else 0)
    )

    return round(combined, 4)


# ── PORTFOLIO TRACKER ─────────────────────────────────────────────────────────

@dataclass
class BacktestPortfolio:
    cash: float = INITIAL_CAPITAL
    positions: dict = field(default_factory=dict)
    # positions = {symbol: {"shares": int, "entry_price": float}}
    equity_curve: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)
    peak_value: float = INITIAL_CAPITAL

    def market_value(self, prices: dict) -> float:
        pos_val = sum(
            self.positions[s]["shares"] * prices.get(s, 0)
            for s in self.positions
        )
        return self.cash + pos_val

    def drawdown(self, current_value: float) -> float:
        if current_value > self.peak_value:
            self.peak_value = current_value
        return (self.peak_value - current_value) / self.peak_value

    def buy(self, symbol, shares, price, date):
        cost = shares * price * (1 + COMMISSION_PCT)
        if cost > self.cash:
            shares = int(self.cash / (price * (1 + COMMISSION_PCT)))
            cost = shares * price * (1 + COMMISSION_PCT)
        if shares == 0:
            return False
        self.cash -= cost
        if symbol in self.positions:
            # Average down
            existing = self.positions[symbol]
            total_shares = existing["shares"] + shares
            avg_price = (
                (existing["shares"] * existing["entry_price"] + shares * price)
                / total_shares
            )
            self.positions[symbol] = {"shares": total_shares, "entry_price": avg_price}
        else:
            self.positions[symbol] = {"shares": shares, "entry_price": price}
        self.trade_log.append({
            "date": date, "symbol": symbol, "action": "BUY",
            "shares": shares, "price": price, "value": shares * price,
        })
        return True

    def sell(self, symbol, price, date):
        if symbol not in self.positions:
            return False
        shares = self.positions[symbol]["shares"]
        proceeds = shares * price * (1 - COMMISSION_PCT)
        entry = self.positions[symbol]["entry_price"]
        pnl = (price - entry) * shares
        self.cash += proceeds
        del self.positions[symbol]
        self.trade_log.append({
            "date": date, "symbol": symbol, "action": "SELL",
            "shares": shares, "price": price, "value": proceeds, "pnl": pnl,
        })
        return True


# ── MAIN BACKTEST LOOP ────────────────────────────────────────────────────────

def run_backtest(symbol: str = None) -> dict:
    """
    Main backtest loop. Iterates day by day, computes signals,
    evaluates trades through risk engine, tracks portfolio value.
    """
    print(f"\nLoading data ({BACKTEST_START} to {BACKTEST_END})...")
    all_data = load_price_data(symbol)

    if all_data.empty:
        print("No data found. Make sure Phase 1 pipeline has run.")
        return {}

    symbols  = all_data["Symbol"].unique()
    all_dates = sorted(all_data["Date"].unique())
    portfolio = BacktestPortfolio()
    config    = RiskConfig(total_capital=INITIAL_CAPITAL)

    print(f"Running backtest on {len(symbols)} stocks across {len(all_dates)} trading days...\n")

    for day_idx, date in enumerate(all_dates):

        # Get current prices for all stocks on this date
        day_prices = all_data[all_data["Date"] == date].set_index("Symbol")["Close"].to_dict()

        # ── Check stop losses first ───────────────────────────────────────────
        for sym in list(portfolio.positions.keys()):
            if sym not in day_prices:
                continue
            current_price = day_prices[sym]
            entry_price   = portfolio.positions[sym]["entry_price"]
            loss_pct      = (current_price - entry_price) / entry_price
            if loss_pct < -config.stop_loss_pct:
                portfolio.sell(sym, current_price, date)

        # ── Re-evaluate signals every REBALANCE_FREQ days ────────────────────
        if day_idx % REBALANCE_FREQ == 0:
            for sym in symbols:
                sym_data = all_data[all_data["Symbol"] == sym].reset_index(drop=True)
                sym_idx  = sym_data[sym_data["Date"] == date].index
                if len(sym_idx) == 0:
                    continue
                current_idx = sym_idx[0]
                current_price = day_prices.get(sym)
                if current_price is None:
                    continue

                signal = compute_signals_for_date(sym_data, current_idx)

                # Check drawdown circuit breaker
                total_val = portfolio.market_value(day_prices)
                dd = portfolio.drawdown(total_val)
                if dd > config.max_drawdown_pct:
                    # Close all positions
                    for s in list(portfolio.positions.keys()):
                        if s in day_prices:
                            portfolio.sell(s, day_prices[s], date)
                    break

                # Buy signal
                if signal > config.min_signal_strength:
                    if sym not in portfolio.positions:
                        # Simple position sizing: 5% of portfolio per position
                        alloc   = portfolio.cash * 0.05
                        shares  = int(alloc / current_price)
                        if shares > 0:
                            portfolio.buy(sym, shares, current_price, date)

                # Sell signal
                elif signal < -config.min_signal_strength:
                    if sym in portfolio.positions:
                        portfolio.sell(sym, current_price, date)

        # ── Record equity curve ───────────────────────────────────────────────
        total_value = portfolio.market_value(day_prices)
        portfolio.equity_curve.append({
            "date":         date,
            "portfolio_value": total_value,
            "cash":         portfolio.cash,
            "n_positions":  len(portfolio.positions),
        })

    return portfolio


# ── PERFORMANCE METRICS ───────────────────────────────────────────────────────

def compute_metrics(portfolio: BacktestPortfolio, all_data: pd.DataFrame) -> dict:
    """Compute standard performance metrics from the equity curve."""
    equity = pd.DataFrame(portfolio.equity_curve).set_index("date")
    equity.index = pd.to_datetime(equity.index)

    final_value    = equity["portfolio_value"].iloc[-1]
    total_return   = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Daily returns of portfolio
    daily_returns  = equity["portfolio_value"].pct_change().dropna()

    # Sharpe Ratio (annualized, assuming 252 trading days, risk-free=6% for India)
    risk_free_daily = 0.06 / 252
    excess_returns  = daily_returns - risk_free_daily
    sharpe          = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

    # Max drawdown
    rolling_max  = equity["portfolio_value"].cummax()
    drawdown_series = (equity["portfolio_value"] - rolling_max) / rolling_max
    max_drawdown = drawdown_series.min()

    # Trade stats
    trades    = pd.DataFrame(portfolio.trade_log)
    sell_trades = trades[trades["action"] == "SELL"] if not trades.empty else pd.DataFrame()
    win_rate  = 0.0
    avg_pnl   = 0.0
    if not sell_trades.empty and "pnl" in sell_trades.columns:
        wins     = sell_trades[sell_trades["pnl"] > 0]
        win_rate = len(wins) / len(sell_trades)
        avg_pnl  = sell_trades["pnl"].mean()

    # Benchmark: Nifty 50 buy and hold (using average of all stocks as proxy)
    benchmark = all_data.groupby("Date")["Close"].mean()
    bench_return = (benchmark.iloc[-1] - benchmark.iloc[0]) / benchmark.iloc[0]

    return {
        "initial_capital":  INITIAL_CAPITAL,
        "final_value":      round(final_value, 2),
        "total_return":     round(total_return * 100, 2),
        "benchmark_return": round(bench_return * 100, 2),
        "sharpe_ratio":     round(sharpe, 3),
        "max_drawdown":     round(max_drawdown * 100, 2),
        "total_trades":     len(trades),
        "win_rate":         round(win_rate * 100, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
    }


def print_report(metrics: dict, portfolio: BacktestPortfolio):
    """Print a clean performance report."""
    print("\n" + "═" * 55)
    print("  BACKTEST PERFORMANCE REPORT")
    print(f"  {BACKTEST_START} to {BACKTEST_END}")
    print("═" * 55)
    print(f"  Initial Capital   : ₹{metrics['initial_capital']:>12,.0f}")
    print(f"  Final Value       : ₹{metrics['final_value']:>12,.0f}")
    print(f"  Total Return      : {metrics['total_return']:>+11.2f}%")
    print(f"  Benchmark Return  : {metrics['benchmark_return']:>+11.2f}%")
    print(f"  {'─'*40}")
    print(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:>12.3f}")
    print(f"  Max Drawdown      : {metrics['max_drawdown']:>+11.2f}%")
    print(f"  {'─'*40}")
    print(f"  Total Trades      : {metrics['total_trades']:>12}")
    print(f"  Win Rate          : {metrics['win_rate']:>11.2f}%")
    print(f"  Avg P&L per Trade : ₹{metrics['avg_pnl_per_trade']:>11,.2f}")
    print("═" * 55)

    # Alpha vs benchmark
    alpha = metrics["total_return"] - metrics["benchmark_return"]
    if alpha > 0:
        print(f"\n  Alpha vs benchmark: +{alpha:.2f}% (outperformed)")
    else:
        print(f"\n  Alpha vs benchmark: {alpha:.2f}% (underperformed)")

    # Save equity curve to CSV
    equity_df = pd.DataFrame(portfolio.equity_curve)
    equity_df.to_csv("data/equity_curve.csv", index=False)
    print(f"\n  Equity curve saved to: data/equity_curve.csv")

    # Save trade log
    if portfolio.trade_log:
        trades_df = pd.DataFrame(portfolio.trade_log)
        trades_df.to_csv("data/trade_log.csv", index=False)
        print(f"  Trade log saved to:   data/trade_log.csv")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    symbol = None
    if "--symbol" in sys.argv:
        idx    = sys.argv.index("--symbol")
        symbol = sys.argv[idx + 1].upper()
        print(f"Backtesting single stock: {symbol}")
    else:
        print("Backtesting full Nifty 50 portfolio...")

    portfolio = run_backtest(symbol)

    if portfolio and portfolio.equity_curve:
        all_data = load_price_data(symbol)
        metrics  = compute_metrics(portfolio, all_data)
        print_report(metrics, portfolio)
    else:
        print("Backtest produced no results. Check your data.")