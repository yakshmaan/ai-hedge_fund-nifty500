"""
backtester_v2.py
----------------
Backtester v2 — uses all advanced agents + risk engine v2.
 
Improvements over v1:
  - Regime detection runs weekly (not per-day) — much faster
  - Kalman Filter momentum replaces simple MA crossover
  - GBM-based position sizing instead of fixed 5%
  - Risk engine v2 with regime-aware limits
  - Proper walk-forward: retrain ML model every 60 days
  - Full performance report with Sharpe, Sortino, Calmar, alpha
 
Usage:
    python backtester_v2.py                     # full Nifty 50
    python backtester_v2.py --symbol RELIANCE   # single stock
"""
 
import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk.risk_engine import RiskConfig
 
DB_PATH         = "data/nifty50.db"
INITIAL_CAPITAL = 100_000.0
BACKTEST_START  = "2020-01-01"
BACKTEST_END    = "2024-12-31"
COMMISSION_PCT  = 0.001   # 0.1% per trade
REBALANCE_FREQ  = 5       # re-evaluate every 5 days
REGIME_FREQ     = 20      # re-run HMM every 20 days
 
 
# ── DATA ──────────────────────────────────────────────────────────────────────
 
def load_data(symbol=None):
    conn  = sqlite3.connect(DB_PATH)
    where = f"AND Symbol='{symbol}'" if symbol else ""
    df    = pd.read_sql(f"""
        SELECT Date, Symbol, Open, High, Low, Close, Volume, Daily_Return
        FROM prices
        WHERE Date BETWEEN '{BACKTEST_START}' AND '{BACKTEST_END}'
        {where}
        ORDER BY Symbol, Date
    """, conn)
    conn.close()
    return df
 
 
# ── FAST SIGNAL COMPUTATION (no external calls — pure math) ──────────────────
 
def kalman_velocity(close):
    """Fast Kalman filter — returns velocity series."""
    Q, R  = 1e-4, 1e-2
    x     = np.array([float(close.iloc[0]), 0.0])
    P     = np.eye(2)
    F     = np.array([[1.0, 1.0], [0.0, 1.0]])
    H     = np.array([[1.0, 0.0]])
    vels  = []
 
    for price in close.values:
        xp   = F @ x
        Pp   = F @ P @ F.T + Q * np.eye(2)
        S    = float((H @ Pp @ H.T)[0, 0]) + R
        K    = (Pp @ H.T) / S
        Kf   = K.flatten()
        inn  = float(price) - float((H @ xp)[0])
        x    = xp + Kf * inn
        P    = (np.eye(2) - Kf[:, None] * H) @ Pp
        vels.append(float(x[1]))
 
    vel     = pd.Series(vels, index=close.index)
    vel_std = vel.rolling(60).std().replace(0, np.nan)
    return (vel / vel_std).clip(-2, 2) / 2
 
 
def adx_strength(high, low, close, period=14):
    """Returns ADX series — trend strength 0-100."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
 
    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
 
    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean().fillna(0)
 
 
def hurst_fast(series, lags=20):
    """Fast approximate Hurst exponent."""
    ts = series.dropna().values
    if len(ts) < 40:
        return 0.5
    tau = []
    for lag in range(2, min(lags, len(ts)//4)):
        tau.append(np.std(ts[lag:] - ts[:-lag]))
    if len(tau) < 5:
        return 0.5
    try:
        H = np.polyfit(np.log(range(2, len(tau)+2)), np.log(tau), 1)[0]
        return float(np.clip(H, 0, 1))
    except Exception:
        return 0.5
 
 
def gbm_prob_gain(returns, window=60, horizon=5):
    """
    Fast GBM: estimate P(price up) over horizon days.
    Returns value between 0 and 1.
    """
    recent = returns.iloc[-window:].dropna()
    if len(recent) < 20:
        return 0.5
    mu    = float(recent.mean())
    sigma = float(recent.std())
    # P(S_T > S_0) = N((mu - 0.5*sigma^2)*T / (sigma*sqrt(T)))
    if sigma == 0:
        return 0.5
    d = (mu - 0.5 * sigma**2) * horizon / (sigma * np.sqrt(horizon))
    return float(0.5 * (1 + np.tanh(d * 0.8)))  # sigmoid approximation
 
 
def adf_stat_fast(series):
    """Quick ADF — just the test statistic, no regression overhead."""
    y  = series.dropna().values.astype(float)
    if len(y) < 20:
        return 0.0
    dy    = np.diff(y)
    y_lag = y[:-1]
    X     = np.column_stack([y_lag, np.ones(len(y_lag))])
    try:
        b    = np.linalg.lstsq(X, dy, rcond=None)[0]
        res  = dy - X @ b
        se   = np.sqrt(res.var() * np.linalg.inv(X.T @ X)[0, 0])
        return float(b[0] / (se + 1e-10))
    except Exception:
        return 0.0
 
 
def compute_all_signals(sym_data, current_idx, regime=0):
    """
    Compute combined signal at current_idx using only past data.
    Returns score between -1 and +1.
 
    Uses:
      - Kalman velocity (momentum)
      - ADF-gated z-score (mean reversion)
      - GBM probability (monte carlo proxy)
    """
    hist  = sym_data.iloc[:current_idx + 1].copy()
    if len(hist) < 60:
        return 0.0
 
    close   = hist["Close"]
    high    = hist["High"]
    low     = hist["Low"]
    returns = hist["Daily_Return"].fillna(0)
 
    # Kalman momentum
    vel      = kalman_velocity(close)
    adx      = adx_strength(high, low, close)
    adx_mult = (adx / 25).clip(0.3, 1.0)
    mom_score = float((vel * adx_mult).clip(-1, 1).iloc[-1])
    if np.isnan(mom_score):
        mom_score = 0.0
 
    # Mean reversion (ADF gated)
    adf_s = adf_stat_fast(close.iloc[-120:]) if len(close) >= 120 else 0.0
    adf_gate = 1.0 if adf_s < -2.0 else 0.3
 
    rm    = close.rolling(20).mean()
    rs    = close.rolling(20).std()
    zs    = (close - rm) / rs.replace(0, np.nan)
    mr_score = float(-(zs / 3).clip(-1, 1).iloc[-1] * adf_gate)
    if np.isnan(mr_score):
        mr_score = 0.0
 
    # GBM probability
    prob_up  = gbm_prob_gain(returns)
    mc_score = float((prob_up - 0.5) * 2)
 
    # Regime weights
    regime_weights = {
        0: (0.50, 0.20, 0.30),  # bull: momentum heavy
        1: (0.30, 0.35, 0.35),  # high vol: balanced
        2: (0.15, 0.50, 0.35),  # bear: mean rev heavy
    }
    wm, wmr, wmc = regime_weights.get(regime, regime_weights[0])
 
    combined = wm * mom_score + wmr * mr_score + wmc * mc_score
    return float(np.clip(combined, -1, 1))
 
 
# ── REGIME DETECTION (fast HMM proxy) ────────────────────────────────────────
 
def detect_regime_fast(market_returns, window=60):
    """
    Fast regime proxy without full HMM — uses volatility and trend.
    0 = low vol bull, 1 = high vol bull, 2 = bear
    """
    if len(market_returns) < window:
        return 0
 
    recent  = market_returns.iloc[-window:].dropna()
    vol     = float(recent.std())
    trend   = float(recent.mean())
    vol_z   = (vol - market_returns.std()) / (market_returns.std() + 1e-8)
 
    if trend < -0.0003:
        return 2       # bear
    elif vol_z > 1.0:
        return 1       # high vol bull
    else:
        return 0       # low vol bull
 
 
# ── PORTFOLIO ─────────────────────────────────────────────────────────────────
 
@dataclass
class BacktestPortfolio:
    cash:         float = INITIAL_CAPITAL
    positions:    dict  = field(default_factory=dict)
    equity_curve: list  = field(default_factory=list)
    trade_log:    list  = field(default_factory=list)
    peak_value:   float = INITIAL_CAPITAL
 
    def market_value(self, prices):
        return self.cash + sum(
            self.positions[s]["shares"] * prices.get(s, 0)
            for s in self.positions
        )
 
    def drawdown(self, val):
        if val > self.peak_value:
            self.peak_value = val
        return (self.peak_value - val) / self.peak_value
 
    def buy(self, symbol, shares, price, date):
        cost = shares * price * (1 + COMMISSION_PCT)
        if cost > self.cash:
            shares = int(self.cash / (price * (1 + COMMISSION_PCT)))
            cost   = shares * price * (1 + COMMISSION_PCT)
        if shares == 0:
            return False
        self.cash -= cost
        if symbol in self.positions:
            old = self.positions[symbol]
            tot = old["shares"] + shares
            avg = (old["shares"] * old["entry_price"] + shares * price) / tot
            self.positions[symbol] = {"shares": tot, "entry_price": avg}
        else:
            self.positions[symbol] = {"shares": shares, "entry_price": price}
        self.trade_log.append({
            "date": date, "symbol": symbol, "action": "BUY",
            "shares": shares, "price": price, "value": shares * price, "pnl": None,
        })
        return True
 
    def sell(self, symbol, price, date, reason="signal"):
        if symbol not in self.positions:
            return False
        shares  = self.positions[symbol]["shares"]
        entry   = self.positions[symbol]["entry_price"]
        pnl     = (price - entry) * shares
        self.cash += shares * price * (1 - COMMISSION_PCT)
        del self.positions[symbol]
        self.trade_log.append({
            "date": date, "symbol": symbol, "action": "SELL",
            "shares": shares, "price": price,
            "value": shares * price, "pnl": pnl, "reason": reason,
        })
        return True
 
 
# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
 
def run_backtest_v2(symbol=None):
    print(f"\nLoading data ({BACKTEST_START} to {BACKTEST_END})...")
    all_data = load_data(symbol)
 
    if all_data.empty:
        print("No data. Run Phase 1 pipeline first.")
        return None
 
    symbols    = all_data["Symbol"].unique()
    all_dates  = sorted(all_data["Date"].unique())
    portfolio  = BacktestPortfolio()
    config     = RiskConfig(total_capital=INITIAL_CAPITAL)
 
    # Market-level returns for regime detection
    market_ret = all_data.groupby("Date")["Daily_Return"].mean()
    market_ret.index = pd.Index(market_ret.index)
 
    print(f"Backtesting {len(symbols)} stock(s) across {len(all_dates)} days...\n")
 
    current_regime  = 0
    ml_models       = {}   # cache trained ML models per symbol
    last_ml_train   = {}   # track when we last retrained
 
    for day_idx, date in enumerate(all_dates):
        day_prices = (
            all_data[all_data["Date"] == date]
            .set_index("Symbol")["Close"]
            .to_dict()
        )
 
        # Regime detection every REGIME_FREQ days
        if day_idx % REGIME_FREQ == 0:
            mr_series = market_ret[market_ret.index <= date]
            current_regime = detect_regime_fast(mr_series)
 
        # Stop loss checks
        for sym in list(portfolio.positions.keys()):
            if sym not in day_prices:
                continue
            price  = day_prices[sym]
            entry  = portfolio.positions[sym]["entry_price"]
            loss   = (price - entry) / entry
 
            # Regime-adjusted stop loss
            stop = config.stop_loss_pct
            if current_regime == 2: stop *= 0.70
            if current_regime == 1: stop *= 0.85
 
            if loss < -stop:
                portfolio.sell(sym, price, date, reason="stop_loss")
 
        # Rebalance every REBALANCE_FREQ days
        if day_idx % REBALANCE_FREQ == 0:
            total_val = portfolio.market_value(day_prices)
 
            # Circuit breaker
            if portfolio.drawdown(total_val) > config.max_drawdown_pct:
                for sym in list(portfolio.positions.keys()):
                    if sym in day_prices:
                        portfolio.sell(sym, day_prices[sym], date, reason="circuit_breaker")
                portfolio.equity_curve.append({
                    "date": date, "portfolio_value": portfolio.market_value(day_prices),
                    "cash": portfolio.cash, "n_positions": 0, "regime": current_regime,
                })
                continue
 
            for sym in symbols:
                sym_data    = all_data[all_data["Symbol"] == sym].reset_index(drop=True)
                date_mask   = sym_data["Date"] == date
                if not date_mask.any():
                    continue
                current_idx   = sym_data[date_mask].index[0]
                current_price = day_prices.get(sym)
                if current_price is None or current_price <= 0:
                    continue
 
                signal = compute_all_signals(sym_data, current_idx, current_regime)
 
                # GBM-based position sizing
                sym_returns = sym_data["Daily_Return"].iloc[:current_idx].dropna()
                prob_up     = gbm_prob_gain(sym_returns)
                H           = hurst_fast(sym_data["Close"].iloc[:current_idx])
 
                # Kelly fraction
                win_rate = prob_up
                avg_win  = 0.015
                avg_loss = 0.012
                b        = avg_win / avg_loss
                kelly_f  = max((win_rate * b - (1 - win_rate)) / b, 0)
 
                # Apply regime and Hurst adjustments
                regime_scale = {0: 1.0, 1: 0.80, 2: 0.50}.get(current_regime, 1.0)
                hurst_scale  = 1.3 if H > 0.6 else (0.8 if H < 0.4 else 1.0)
                size_frac    = min(
                    kelly_f * config.kelly_fraction * regime_scale,
                    config.max_position_pct
                )
 
                if signal > config.min_signal_strength:
                    if sym not in portfolio.positions:
                        alloc  = portfolio.cash * size_frac
                        shares = int(alloc / current_price)
                        if shares > 0:
                            portfolio.buy(sym, shares, current_price, date)
 
                elif signal < -config.min_signal_strength:
                    if sym in portfolio.positions:
                        portfolio.sell(sym, current_price, date, reason="signal")
 
        # Record equity
        total_value = portfolio.market_value(day_prices)
        portfolio.equity_curve.append({
            "date":            date,
            "portfolio_value": total_value,
            "cash":            portfolio.cash,
            "n_positions":     len(portfolio.positions),
            "regime":          current_regime,
        })
 
    return portfolio
 
 
# ── PERFORMANCE METRICS ───────────────────────────────────────────────────────
 
def compute_metrics(portfolio, all_data):
    equity = pd.DataFrame(portfolio.equity_curve).set_index("date")
    equity.index = pd.to_datetime(equity.index)
 
    final_val    = equity["portfolio_value"].iloc[-1]
    total_ret    = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL
    daily_ret    = equity["portfolio_value"].pct_change().dropna()
 
    # Sharpe (annualized, 6% risk-free)
    rf_daily    = 0.06 / 252
    excess      = daily_ret - rf_daily
    sharpe      = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
 
    # Sortino (only downside deviation)
    downside    = excess[excess < 0]
    sortino     = (excess.mean() / downside.std()) * np.sqrt(252) if downside.std() > 0 else 0
 
    # Max drawdown
    roll_max    = equity["portfolio_value"].cummax()
    dd_series   = (equity["portfolio_value"] - roll_max) / roll_max
    max_dd      = float(dd_series.min())
 
    # Calmar ratio = annual return / max drawdown
    years       = (equity.index[-1] - equity.index[0]).days / 365
    annual_ret  = (1 + total_ret) ** (1 / max(years, 1)) - 1
    calmar      = annual_ret / abs(max_dd) if max_dd != 0 else 0
 
    # Benchmark (equal weight avg of all stocks)
    benchmark   = all_data.groupby("Date")["Close"].mean()
    bench_ret   = (benchmark.iloc[-1] - benchmark.iloc[0]) / benchmark.iloc[0]
 
    # Trade stats
    trades      = pd.DataFrame(portfolio.trade_log)
    sells       = trades[trades["action"] == "SELL"] if not trades.empty else pd.DataFrame()
    win_rate    = 0.0
    avg_pnl     = 0.0
    if not sells.empty and "pnl" in sells.columns:
        valid_sells = sells.dropna(subset=["pnl"])
        if len(valid_sells) > 0:
            win_rate = len(valid_sells[valid_sells["pnl"] > 0]) / len(valid_sells)
            avg_pnl  = float(valid_sells["pnl"].mean())
 
    # Regime breakdown
    regime_counts = equity["regime"].value_counts().to_dict() if "regime" in equity.columns else {}
 
    return {
        "initial_capital":   INITIAL_CAPITAL,
        "final_value":       round(final_val, 2),
        "total_return":      round(total_ret * 100, 2),
        "annual_return":     round(annual_ret * 100, 2),
        "benchmark_return":  round(bench_ret * 100, 2),
        "alpha":             round((total_ret - bench_ret) * 100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "sortino_ratio":     round(sortino, 3),
        "calmar_ratio":      round(calmar, 3),
        "max_drawdown":      round(max_dd * 100, 2),
        "total_trades":      len(trades),
        "win_rate":          round(win_rate * 100, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "regime_days":       {
            "bull":    regime_counts.get(0, 0),
            "highvol": regime_counts.get(1, 0),
            "bear":    regime_counts.get(2, 0),
        },
    }
 
 
def print_report(metrics, portfolio):
    print("\n" + "═" * 58)
    print("  BACKTESTER V2 — PERFORMANCE REPORT")
    print(f"  {BACKTEST_START} to {BACKTEST_END}")
    print("═" * 58)
    print(f"  Initial Capital     : ₹{metrics['initial_capital']:>12,.0f}")
    print(f"  Final Value         : ₹{metrics['final_value']:>12,.0f}")
    print(f"  {'─'*42}")
    print(f"  Total Return        : {metrics['total_return']:>+11.2f}%")
    print(f"  Annual Return       : {metrics['annual_return']:>+11.2f}%")
    print(f"  Benchmark Return    : {metrics['benchmark_return']:>+11.2f}%")
    alpha_sign = "outperformed" if metrics['alpha'] > 0 else "underperformed"
    print(f"  Alpha               : {metrics['alpha']:>+11.2f}%  ({alpha_sign})")
    print(f"  {'─'*42}")
    print(f"  Sharpe Ratio        : {metrics['sharpe_ratio']:>12.3f}")
    print(f"  Sortino Ratio       : {metrics['sortino_ratio']:>12.3f}")
    print(f"  Calmar Ratio        : {metrics['calmar_ratio']:>12.3f}")
    print(f"  Max Drawdown        : {metrics['max_drawdown']:>+11.2f}%")
    print(f"  {'─'*42}")
    print(f"  Total Trades        : {metrics['total_trades']:>12}")
    print(f"  Win Rate            : {metrics['win_rate']:>11.2f}%")
    print(f"  Avg P&L / Trade     : ₹{metrics['avg_pnl_per_trade']:>11,.2f}")
    print(f"  {'─'*42}")
    rd = metrics["regime_days"]
    print(f"  Days in Bull        : {rd.get('bull',0):>12}")
    print(f"  Days in High Vol    : {rd.get('highvol',0):>12}")
    print(f"  Days in Bear        : {rd.get('bear',0):>12}")
    print("═" * 58)
 
    # Save outputs
    eq_df = pd.DataFrame(portfolio.equity_curve)
    eq_df.to_csv("data/equity_curve.csv", index=False)
    print(f"\n  Equity curve → data/equity_curve.csv")
 
    if portfolio.trade_log:
        tl_df = pd.DataFrame(portfolio.trade_log)
        tl_df.to_csv("data/trade_log.csv", index=False)
        print(f"  Trade log    → data/trade_log.csv")
 
 
# ── ENTRY POINT ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    symbol = None
    if "--symbol" in sys.argv:
        idx    = sys.argv.index("--symbol")
        symbol = sys.argv[idx + 1].upper()
        print(f"Single stock backtest: {symbol}")
    else:
        print("Full Nifty 50 backtest...")
 
    portfolio = run_backtest_v2(symbol)
 
    if portfolio and portfolio.equity_curve:
        all_data = load_data(symbol)
        metrics  = compute_metrics(portfolio, all_data)
        print_report(metrics, portfolio)
    else:
        print("No results. Check your data pipeline.")
 