"""
central_command_v2.py
---------------------
Central Command V2 — Maximum Returns with Early Warning Exit.
 
Strategy:
  - Default: 100% in V5 passive (all 48 stocks, equal weight)
  - Early warning system monitors 3 signals BEFORE crash hits:
      1. Momentum deterioration: 10d return turns negative after bull run
      2. Volatility spike: rolling vol jumps 2x above 60d average
      3. Breadth collapse: >60% of stocks in portfolio trading below 20d MA
  - When 2 of 3 signals trigger → exit to 100% cash immediately
  - When all 3 clear → re-enter V5 passive
  - No V4 trading, no signal computation, no commission on active trades
  - Only cost is entry/exit of passive basket
 
Target: +190-220% return, -15 to -20% max drawdown
 
Usage:
    python central_command_v2.py
"""
 
import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass, field
 
DB_PATH         = "data/nifty50.db"
INITIAL_CAPITAL = 100_000.0
BACKTEST_START  = "2020-01-01"
BACKTEST_END    = "2024-12-01"
COMMISSION_PCT  = 0.001
INDEX_TOP_N     = 48
 
# ── CC2 EARLY WARNING PARAMETERS ──────────────────────────────────────────────
MOMENTUM_WINDOW     = 10    # days to measure recent momentum
VOL_WINDOW          = 60    # baseline volatility window
VOL_SPIKE_MULT      = 2.0   # vol must be 2x baseline to trigger
BREADTH_THRESHOLD   = 0.60  # 60% stocks below 20d MA triggers warning
MA_WINDOW           = 20    # moving average window for breadth
SIGNALS_TO_EXIT     = 2     # number of signals needed to exit (out of 3)
SIGNALS_TO_REENTER  = 0     # all signals must clear to re-enter
COOLDOWN_DAYS       = 20    # min days in cash before re-entering
# ──────────────────────────────────────────────────────────────────────────────
 
 
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
    avg_vol = df.groupby("Symbol")["Volume"].mean().nlargest(n)
    return list(avg_vol.index)
 
 
def compute_early_warning(all_data, top_syms, date, all_dates):
    """
    Compute 3 early warning signals.
    Returns (n_signals_triggered, signal_details)
    """
    hist = all_data[all_data["Date"] <= date]
    market_ret = hist.groupby("Date")["Daily_Return"].mean().dropna()
 
    if len(market_ret) < VOL_WINDOW + MOMENTUM_WINDOW:
        return 0, {}
 
    signals = {}
 
    # ── Signal 1: Momentum deterioration ──────────────────────────────────
    # 10d return turns negative
    recent_10d = market_ret.iloc[-MOMENTUM_WINDOW:].sum()
    prior_10d  = market_ret.iloc[-MOMENTUM_WINDOW*2:-MOMENTUM_WINDOW].sum()
    mom_trigger = (recent_10d < 0) and (prior_10d > 0)  # was positive, now negative
    signals["momentum"] = mom_trigger
 
    # ── Signal 2: Volatility spike ─────────────────────────────────────────
    # Current vol > 2x baseline vol
    baseline_vol = market_ret.iloc[-VOL_WINDOW:-MOMENTUM_WINDOW].std()
    current_vol  = market_ret.iloc[-MOMENTUM_WINDOW:].std()
    vol_trigger  = current_vol > (VOL_SPIKE_MULT * baseline_vol) if baseline_vol > 0 else False
    signals["volatility"] = vol_trigger
 
    # ── Signal 3: Breadth collapse ─────────────────────────────────────────
    # >60% of top stocks trading below their 20d MA
    below_ma = 0
    total    = 0
    for sym in top_syms:
        sym_hist = hist[hist["Symbol"] == sym]["Close"]
        if len(sym_hist) < MA_WINDOW: continue
        ma    = sym_hist.iloc[-MA_WINDOW:].mean()
        price = sym_hist.iloc[-1]
        total += 1
        if price < ma:
            below_ma += 1
    breadth_ratio   = below_ma / total if total > 0 else 0
    breadth_trigger = breadth_ratio > BREADTH_THRESHOLD
    signals["breadth"] = breadth_trigger
 
    n_triggered = sum(signals.values())
    return n_triggered, signals
 
 
@dataclass
class CC2Portfolio:
    cash:         float = INITIAL_CAPITAL
    positions:    dict  = field(default_factory=dict)
    equity_curve: list  = field(default_factory=list)
    trade_log:    list  = field(default_factory=list)
    peak_value:   float = INITIAL_CAPITAL
    mode:         str   = "V5"       # "V5" or "CASH"
    mode_log:     list  = field(default_factory=list)
    days_in_cash: int   = 0
 
    def market_value(self, prices):
        return self.cash + sum(
            self.positions[s]["shares"] * prices.get(s, 0)
            for s in self.positions
        )
 
    def drawdown(self, val):
        if val > self.peak_value: self.peak_value = val
        return (self.peak_value - val) / self.peak_value
 
    def buy(self, symbol, shares, price, date):
        cost = shares * price * (1 + COMMISSION_PCT)
        if cost > self.cash:
            shares = int(self.cash / (price * (1 + COMMISSION_PCT)))
            cost   = shares * price * (1 + COMMISSION_PCT)
        if shares == 0: return False
        self.cash -= cost
        if symbol in self.positions:
            old = self.positions[symbol]; tot = old["shares"] + shares
            avg = (old["shares"] * old["entry_price"] + shares * price) / tot
            self.positions[symbol] = {"shares": tot, "entry_price": avg}
        else:
            self.positions[symbol] = {"shares": shares, "entry_price": price}
        self.trade_log.append({
            "date": date, "symbol": symbol, "action": "BUY",
            "shares": shares, "price": price,
            "value": shares * price, "pnl": None, "mode": self.mode,
        })
        return True
 
    def sell(self, symbol, price, date, reason="exit"):
        if symbol not in self.positions: return False
        shares = self.positions[symbol]["shares"]
        entry  = self.positions[symbol]["entry_price"]
        pnl    = (price - entry) * shares
        self.cash += shares * price * (1 - COMMISSION_PCT)
        del self.positions[symbol]
        self.trade_log.append({
            "date": date, "symbol": symbol, "action": "SELL",
            "shares": shares, "price": price,
            "value": shares * price, "pnl": pnl,
            "reason": reason, "mode": self.mode,
        })
        return True
 
    def enter_v5(self, top_symbols, day_prices, date):
        per_stock = self.cash / len(top_symbols)
        for sym in top_symbols:
            price = day_prices.get(sym)
            if price and price > 0:
                shares = int(per_stock / (price * (1 + COMMISSION_PCT)))
                if shares > 0:
                    self.buy(sym, shares, price, date)
        self.mode = "V5"
        self.days_in_cash = 0
        self.mode_log.append({"date": date, "mode": "V5", "reason": "signals_cleared"})
 
    def exit_to_cash(self, day_prices, date, reason="warning"):
        for sym in list(self.positions.keys()):
            if sym in day_prices:
                self.sell(sym, day_prices[sym], date, reason=reason)
        self.mode = "CASH"
        self.mode_log.append({"date": date, "mode": "CASH", "reason": reason})
 
 
def run_cc2():
    print(f"\nCentral Command V2 ({BACKTEST_START} to {BACKTEST_END})...")
    all_data  = load_data()
    all_dates = sorted(all_data["Date"].unique())
    portfolio = CC2Portfolio()
    top_syms  = get_top_n_symbols(all_data, INDEX_TOP_N)
 
    print(f"Backtesting 48 stocks across {len(all_dates)} days...")
    print(f"  Strategy: V5 passive + early warning exit to cash")
    print(f"  Exit on {SIGNALS_TO_EXIT}/3 signals | Cooldown: {COOLDOWN_DAYS}d\n")
 
    switches = 0
 
    # Start in V5
    first_prices = all_data[all_data["Date"] == all_dates[0]].set_index("Symbol")["Close"].to_dict()
    portfolio.enter_v5(top_syms, first_prices, all_dates[0])
 
    for day_idx, date in enumerate(all_dates):
        day_prices = all_data[all_data["Date"] == date].set_index("Symbol")["Close"].to_dict()
 
        # Track cash days
        if portfolio.mode == "CASH":
            portfolio.days_in_cash += 1
 
        # Check early warning every 5 days
        if day_idx % 5 == 0 and day_idx > VOL_WINDOW + MOMENTUM_WINDOW:
            n_signals, signal_details = compute_early_warning(all_data, top_syms, date, all_dates)
 
            # ── Exit V5 → Cash ─────────────────────────────────────────────
            if portfolio.mode == "V5" and n_signals >= SIGNALS_TO_EXIT:
                triggers = [k for k, v in signal_details.items() if v]
                print(f"  [{date}] EXIT to cash — signals: {triggers}")
                portfolio.exit_to_cash(day_prices, date, reason="+".join(triggers))
                switches += 1
 
            # ── Re-enter V5 from Cash ──────────────────────────────────────
            elif (portfolio.mode == "CASH"
                  and n_signals <= SIGNALS_TO_REENTER
                  and portfolio.days_in_cash >= COOLDOWN_DAYS):
                print(f"  [{date}] RE-ENTER V5 — all signals cleared")
                portfolio.enter_v5(top_syms, day_prices, date)
                switches += 1
 
        total_value = portfolio.market_value(day_prices)
        dd = portfolio.drawdown(total_value)
 
        # Emergency exit if drawdown hits 20% even in V5
        if portfolio.mode == "V5" and dd >= 0.20:
            print(f"  [{date}] EMERGENCY EXIT — DD={dd*100:.1f}%")
            portfolio.exit_to_cash(day_prices, date, reason="emergency_dd")
            switches += 1
 
        portfolio.equity_curve.append({
            "date":            date,
            "portfolio_value": total_value,
            "cash":            portfolio.cash,
            "n_positions":     len(portfolio.positions),
            "mode":            portfolio.mode,
            "drawdown":        -dd,
        })
 
    print(f"\n  Total switches: {switches}")
    return portfolio
 
 
def compute_metrics(portfolio, all_data):
    equity       = pd.DataFrame(portfolio.equity_curve).set_index("date")
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
    benchmark    = all_data.groupby("Date")["Close"].mean()
    bench_ret    = (benchmark.iloc[-1] - benchmark.iloc[0]) / benchmark.iloc[0]
    bench_roll   = benchmark.cummax()
    bench_max_dd = float(((benchmark - bench_roll) / bench_roll).min())
    trades       = pd.DataFrame(portfolio.trade_log)
    sells        = trades[trades["action"] == "SELL"] if not trades.empty else pd.DataFrame()
    win_rate     = 0.0; avg_pnl = 0.0
    if not sells.empty and "pnl" in sells.columns:
        valid = sells.dropna(subset=["pnl"])
        if len(valid) > 0:
            win_rate = len(valid[valid["pnl"] > 0]) / len(valid)
            avg_pnl  = float(valid["pnl"].mean())
    mode_counts = equity["mode"].value_counts().to_dict() if "mode" in equity.columns else {}
 
    return {
        "initial_capital":   INITIAL_CAPITAL,
        "final_value":       round(final_val, 2),
        "total_return":      round(total_ret * 100, 2),
        "annual_return":     round(annual * 100, 2),
        "benchmark_return":  round(bench_ret * 100, 2),
        "alpha":             round((total_ret - bench_ret) * 100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "sortino_ratio":     round(sortino, 3),
        "calmar_ratio":      round(calmar, 3),
        "max_drawdown":      round(max_dd * 100, 2),
        "bench_max_dd":      round(bench_max_dd * 100, 2),
        "total_trades":      len(trades),
        "win_rate":          round(win_rate * 100, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "days_in_v5":        mode_counts.get("V5", 0),
        "days_in_cash":      mode_counts.get("CASH", 0),
    }
 
 
def print_report(metrics, portfolio):
    print("\n" + "═" * 58)
    print("  CENTRAL COMMAND V2 — EARLY WARNING MAX RETURNS")
    print(f"  {BACKTEST_START} to {BACKTEST_END}")
    print("=" * 58)
    print(f"  Initial Capital     : ₹{metrics['initial_capital']:>12,.0f}")
    print(f"  Final Value         : ₹{metrics['final_value']:>12,.0f}")
    print(f"  {'─'*42}")
    print(f"  Total Return        : {metrics['total_return']:>+11.2f}%")
    print(f"  Annual Return       : {metrics['annual_return']:>+11.2f}%")
    print(f"  Benchmark Return    : {metrics['benchmark_return']:>+11.2f}%")
    print(f"  Alpha               : {metrics['alpha']:>+11.2f}%  "
          f"({'outperformed' if metrics['alpha'] > 0 else 'underperformed'})")
    print(f"  {'─'*42}")
    print(f"  Sharpe Ratio        : {metrics['sharpe_ratio']:>12.3f}")
    print(f"  Sortino Ratio       : {metrics['sortino_ratio']:>12.3f}")
    print(f"  Calmar Ratio        : {metrics['calmar_ratio']:>12.3f}")
    print(f"  {'─'*42}")
    print(f"  CC2 Max Drawdown    : {metrics['max_drawdown']:>+11.2f}%")
    print(f"  Benchmark Max DD    : {metrics['bench_max_dd']:>+11.2f}%")
    print(f"  {'─'*42}")
    print(f"  Total Trades        : {metrics['total_trades']:>12}")
    print(f"  Win Rate            : {metrics['win_rate']:>11.2f}%")
    print(f"  Avg P&L / Trade     : ₹{metrics['avg_pnl_per_trade']:>11,.2f}")
    print(f"  {'─'*42}")
    print(f"  Days in V5 passive  : {metrics['days_in_v5']:>12}")
    print(f"  Days in Cash        : {metrics['days_in_cash']:>12}")
    print("=" * 58)
    print("\n  ── Full Comparison ─────────────────────────────────")
    print(f"  {'Metric':<22} {'V4':>8} {'CC2':>8} {'CC1':>8} {'V6':>8} {'V5':>8} {'Bench':>8}")
    print(f"  {'─'*68}")
    print(f"  {'Total Return':<22} {'+66.34%':>8} {metrics['total_return']:>+7.2f}% {'+64.46%':>8} {'+110.40%':>8} {'+204.47%':>8} {'+176.48%':>8}")
    print(f"  {'Max Drawdown':<22} {'-16.19%':>8} {metrics['max_drawdown']:>+7.2f}% {'-30.16%':>8} {'-21.74%':>8} {'-40.63%':>8} {'-34.82%':>8}")
    print("=" * 58)
 
    pd.DataFrame(portfolio.equity_curve).to_csv("data/equity_curve_cc2.csv", index=False)
    print(f"\n  Equity curve → data/equity_curve_cc2.csv")
    if portfolio.trade_log:
        pd.DataFrame(portfolio.trade_log).to_csv("data/trade_log_cc2.csv", index=False)
        print(f"  Trade log    → data/trade_log_cc2.csv")
    if portfolio.mode_log:
        pd.DataFrame(portfolio.mode_log).to_csv("data/mode_log_cc2.csv", index=False)
        print(f"  Mode log     → data/mode_log_cc2.csv")
 
 
if __name__ == "__main__":
    print("Central Command V2 — Early Warning Maximum Returns")
    portfolio = run_cc2()
    if portfolio and portfolio.equity_curve:
        all_data = load_data()
        metrics  = compute_metrics(portfolio, all_data)
        print_report(metrics, portfolio)
    else:
        print("No results.")
 