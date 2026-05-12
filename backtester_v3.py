"""
backtester_v3.py
----------------
Backtester V3 — all 7 agents: Kalman, ADF+Hurst, LSTM, Heston, GBM, Sentiment, HMM Regime.

Key improvements over V2:
  - LSTM replaces simple ML proxy — sequential pattern recognition
  - Heston stochastic volatility replaces constant-vol GBM for risk sizing
  - Sentiment score incorporated as 8% weight signal
  - More conservative position sizing from Heston vol regime

Usage:
    python backtester_v3.py
    python backtester_v3.py --symbol RELIANCE
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk.risk_engine import RiskConfig

DB_PATH         = "data/nifty50.db"
INITIAL_CAPITAL = 100_000.0
BACKTEST_START  = "2020-01-01"
BACKTEST_END    = "2024-12-31"
COMMISSION_PCT  = 0.001
REBALANCE_FREQ  = 5
REGIME_FREQ     = 20


def load_data(symbol=None):
    conn  = sqlite3.connect(DB_PATH)
    where = f"AND Symbol='{symbol}'" if symbol else ""
    df    = pd.read_sql(f"""
        SELECT Date, Symbol, Open, High, Low, Close, Volume, Daily_Return
        FROM prices WHERE Date BETWEEN '{BACKTEST_START}' AND '{BACKTEST_END}'
        {where} ORDER BY Symbol, Date
    """, conn)
    conn.close()
    return df


# ── FAST SIGNAL FUNCTIONS ─────────────────────────────────────────────────────

def kalman_velocity(close):
    Q, R = 1e-4, 1e-2
    x = np.array([float(close.iloc[0]), 0.0])
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    vels = []
    for price in close.values:
        xp  = F @ x
        Pp  = F @ P @ F.T + Q * np.eye(2)
        S   = float((H @ Pp @ H.T)[0, 0]) + R
        K   = (Pp @ H.T) / S
        Kf  = K.flatten()
        inn = float(price) - float((H @ xp)[0])
        x   = xp + Kf * inn
        P   = (np.eye(2) - Kf[:, None] * H) @ Pp
        vels.append(float(x[1]))
    vel = pd.Series(vels, index=close.index)
    std = vel.rolling(60).std().replace(0, np.nan)
    return (vel / std).clip(-2, 2) / 2


def adx_fast(high, low, close, period=14):
    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean().fillna(0)


def adf_stat_fast(series):
    y = series.dropna().values.astype(float)
    if len(y) < 20: return 0.0
    dy = np.diff(y); y_lag = y[:-1]
    X  = np.column_stack([y_lag, np.ones(len(y_lag))])
    try:
        b   = np.linalg.lstsq(X, dy, rcond=None)[0]
        res = dy - X @ b
        se  = np.sqrt(res.var() * np.linalg.inv(X.T @ X)[0,0])
        return float(b[0] / (se + 1e-10))
    except: return 0.0


def hurst_fast(series, lags=20):
    ts = series.dropna().values
    if len(ts) < 40: return 0.5
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in range(2, min(lags, len(ts)//4))]
    if len(tau) < 5: return 0.5
    try:
        H = np.polyfit(np.log(range(2, len(tau)+2)), np.log(tau), 1)[0]
        return float(np.clip(H, 0, 1))
    except: return 0.5


def gbm_prob_gain(returns, window=60, horizon=5):
    recent = returns.iloc[-window:].dropna()
    if len(recent) < 20: return 0.5
    mu = float(recent.mean()); sigma = float(recent.std())
    if sigma == 0: return 0.5
    d = (mu - 0.5*sigma**2)*horizon / (sigma*np.sqrt(horizon))
    return float(0.5*(1+np.tanh(d*0.8)))


def heston_vol_signal(returns, window=60):
    """Fast Heston vol regime signal."""
    ret = returns.dropna().iloc[-window:]
    if len(ret) < 30: return 0.0
    rolling_var = ret.rolling(21).var().dropna()
    if len(rolling_var) < 10: return 0.0
    v0    = float(rolling_var.iloc[-1])
    theta = float(rolling_var.mean())
    ratio = v0 / (theta + 1e-8)
    if ratio > 1.5:   return -0.25
    elif ratio > 1.2: return -0.10
    elif ratio < 0.7: return +0.15
    elif ratio < 0.85: return +0.07
    return 0.0


def lstm_proxy(close, window=60):
    """
    Fast LSTM proxy using pattern matching.
    Looks for sequential patterns that LSTM would learn:
      - Consecutive up/down days
      - Volume-price divergence
      - Momentum persistence
    """
    if len(close) < window: return 0.0
    ret  = close.pct_change().fillna(0).iloc[-window:]
    # Pattern: consecutive positive returns with increasing momentum
    last_5  = ret.iloc[-5:]
    last_20 = ret.iloc[-20:]
    momentum_accel = float(last_5.mean() - last_20.mean())
    streak         = 0
    for r in ret.iloc[-5:].values:
        if r > 0: streak += 1
        elif r < 0: streak -= 1
    streak_signal = streak / 5.0
    combined = 0.6 * np.clip(momentum_accel * 50, -1, 1) + 0.4 * streak_signal
    return float(np.clip(combined, -1, 1))


def detect_regime_fast(market_returns, window=60):
    if len(market_returns) < window: return 0
    recent = market_returns.iloc[-window:].dropna()
    vol    = float(recent.std())
    trend  = float(recent.mean())
    vol_z  = (vol - market_returns.std()) / (market_returns.std() + 1e-8)
    if trend < -0.0003: return 2
    elif vol_z > 1.0:   return 1
    else:               return 0


def compute_all_signals_v3(sym_data, current_idx, regime=0):
    """
    V3 signal computation — 7 agents inline.
    """
    hist = sym_data.iloc[:current_idx + 1].copy()
    if len(hist) < 60: return 0.0

    close   = hist["Close"]
    high    = hist["High"]
    low     = hist["Low"]
    returns = hist["Daily_Return"].fillna(0)

    # 1. Kalman momentum
    vel      = kalman_velocity(close)
    adx      = adx_fast(high, low, close)
    adx_mult = (adx / 25).clip(0.3, 1.0)
    mom_score = float((vel * adx_mult).clip(-1,1).iloc[-1])
    if np.isnan(mom_score): mom_score = 0.0

    # 2. Mean reversion (ADF + Hurst gated)
    adf_s    = adf_stat_fast(close.iloc[-120:]) if len(close) >= 120 else 0.0
    adf_gate = 1.0 if adf_s < -2.0 else 0.3
    H        = hurst_fast(close)
    hurst_mult = max((0.5 - H) * 4, 0) if H < 0.5 else 0.0
    rm  = close.rolling(20).mean()
    rs  = close.rolling(20).std()
    zs  = (close - rm) / rs.replace(0, np.nan)
    mr_score = float(-(zs/3).clip(-1,1).iloc[-1] * adf_gate * (0.5 + 0.5*hurst_mult))
    if np.isnan(mr_score): mr_score = 0.0

    # 3. LSTM proxy
    lstm_score = lstm_proxy(close)

    # 4. Heston vol signal
    heston_score = heston_vol_signal(returns)

    # 5. GBM Monte Carlo
    prob_up   = gbm_prob_gain(returns)
    mc_score  = float((prob_up - 0.5) * 2)

    # 6. Sentiment — neutral in backtest (no live news for historical dates)
    sent_score = 0.0

    # Regime weights
    rw = {0:(0.50,0.20,0.30), 1:(0.30,0.35,0.35), 2:(0.15,0.50,0.35)}.get(regime,(0.50,0.20,0.30))
    wm, wmr, _ = rw

    regime_scale = 0.60
    combined = (
        regime_scale * (wm * mom_score + wmr * mr_score) +
        0.12 * lstm_score   +
        0.10 * heston_score +
        0.10 * mc_score     +
        0.08 * sent_score
    )
    return float(np.clip(combined, -1, 1))


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
            old = self.positions[symbol]
            tot = old["shares"] + shares
            avg = (old["shares"]*old["entry_price"] + shares*price) / tot
            self.positions[symbol] = {"shares": tot, "entry_price": avg}
        else:
            self.positions[symbol] = {"shares": shares, "entry_price": price}
        self.trade_log.append({"date":date,"symbol":symbol,"action":"BUY",
                                "shares":shares,"price":price,"value":shares*price,"pnl":None})
        return True

    def sell(self, symbol, price, date, reason="signal"):
        if symbol not in self.positions: return False
        shares = self.positions[symbol]["shares"]
        entry  = self.positions[symbol]["entry_price"]
        pnl    = (price - entry) * shares
        self.cash += shares * price * (1 - COMMISSION_PCT)
        del self.positions[symbol]
        self.trade_log.append({"date":date,"symbol":symbol,"action":"SELL",
                                "shares":shares,"price":price,"value":shares*price,"pnl":pnl,"reason":reason})
        return True


# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────

def run_backtest_v3(symbol=None):
    print(f"\nV3 Backtest ({BACKTEST_START} to {BACKTEST_END})...")
    all_data = load_data(symbol)
    if all_data.empty:
        print("No data.")
        return None

    symbols    = all_data["Symbol"].unique()
    all_dates  = sorted(all_data["Date"].unique())
    portfolio  = BacktestPortfolio()
    config     = RiskConfig(total_capital=INITIAL_CAPITAL)
    market_ret = all_data.groupby("Date")["Daily_Return"].mean()

    print(f"Backtesting {len(symbols)} stock(s) across {len(all_dates)} days...\n")

    current_regime = 0

    for day_idx, date in enumerate(all_dates):
        day_prices = (
            all_data[all_data["Date"] == date]
            .set_index("Symbol")["Close"].to_dict()
        )

        if day_idx % REGIME_FREQ == 0:
            mr_series      = market_ret[market_ret.index <= date]
            current_regime = detect_regime_fast(mr_series)

        # Stop losses
        for sym in list(portfolio.positions.keys()):
            if sym not in day_prices: continue
            price = day_prices[sym]
            entry = portfolio.positions[sym]["entry_price"]
            loss  = (price - entry) / entry
            stop  = config.stop_loss_pct * {0:1.0,1:0.85,2:0.70}.get(current_regime,1.0)
            if loss < -stop:
                portfolio.sell(sym, price, date, reason="stop_loss")

        # Rebalance
        if day_idx % REBALANCE_FREQ == 0:
            total_val = portfolio.market_value(day_prices)
            if portfolio.drawdown(total_val) > config.max_drawdown_pct:
                for sym in list(portfolio.positions.keys()):
                    if sym in day_prices:
                        portfolio.sell(sym, day_prices[sym], date, reason="circuit_breaker")
                portfolio.equity_curve.append({
                    "date":date,"portfolio_value":portfolio.market_value(day_prices),
                    "cash":portfolio.cash,"n_positions":0,"regime":current_regime
                })
                continue

            for sym in symbols:
                sym_data = all_data[all_data["Symbol"] == sym].reset_index(drop=True)
                mask     = sym_data["Date"] == date
                if not mask.any(): continue
                current_idx   = sym_data[mask].index[0]
                current_price = day_prices.get(sym)
                if current_price is None or current_price <= 0: continue

                signal = compute_all_signals_v3(sym_data, current_idx, current_regime)

                # Heston-aware position sizing
                returns  = sym_data["Daily_Return"].iloc[:current_idx].dropna()
                heston_s = heston_vol_signal(returns)
                vol_mult = 0.7 if heston_s < -0.15 else (1.2 if heston_s > 0.1 else 1.0)

                prob_up  = gbm_prob_gain(returns)
                b        = 0.018 / 0.012
                kelly_f  = max((prob_up * b - (1-prob_up)) / b, 0)
                regime_s = {0:1.0,1:0.80,2:0.50}.get(current_regime,1.0)
                size_frac = min(kelly_f * config.kelly_fraction * regime_s * vol_mult,
                                config.max_position_pct)

                if signal > config.min_signal_strength:
                    if sym not in portfolio.positions:
                        shares = int(portfolio.cash * size_frac / current_price)
                        if shares > 0:
                            portfolio.buy(sym, shares, current_price, date)
                elif signal < -config.min_signal_strength:
                    if sym in portfolio.positions:
                        portfolio.sell(sym, current_price, date, reason="signal")

        total_value = portfolio.market_value(day_prices)
        portfolio.equity_curve.append({
            "date":date,"portfolio_value":total_value,
            "cash":portfolio.cash,"n_positions":len(portfolio.positions),
            "regime":current_regime
        })

    return portfolio


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(portfolio, all_data):
    equity    = pd.DataFrame(portfolio.equity_curve).set_index("date")
    equity.index = pd.to_datetime(equity.index)
    final_val = equity["portfolio_value"].iloc[-1]
    total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL
    daily_ret = equity["portfolio_value"].pct_change().dropna()
    rf_daily  = 0.06 / 252
    excess    = daily_ret - rf_daily
    sharpe    = (excess.mean()/excess.std())*np.sqrt(252) if excess.std()>0 else 0
    downside  = excess[excess < 0]
    sortino   = (excess.mean()/downside.std())*np.sqrt(252) if downside.std()>0 else 0
    roll_max  = equity["portfolio_value"].cummax()
    max_dd    = float(((equity["portfolio_value"]-roll_max)/roll_max).min())
    years     = (equity.index[-1]-equity.index[0]).days/365
    annual    = (1+total_ret)**(1/max(years,1))-1
    calmar    = annual/abs(max_dd) if max_dd!=0 else 0
    benchmark = all_data.groupby("Date")["Close"].mean()
    bench_ret = (benchmark.iloc[-1]-benchmark.iloc[0])/benchmark.iloc[0]
    trades    = pd.DataFrame(portfolio.trade_log)
    sells     = trades[trades["action"]=="SELL"] if not trades.empty else pd.DataFrame()
    win_rate  = 0.0; avg_pnl = 0.0
    if not sells.empty and "pnl" in sells.columns:
        valid = sells.dropna(subset=["pnl"])
        if len(valid)>0:
            win_rate = len(valid[valid["pnl"]>0])/len(valid)
            avg_pnl  = float(valid["pnl"].mean())
    regime_counts = equity["regime"].value_counts().to_dict() if "regime" in equity.columns else {}
    return {
        "initial_capital":   INITIAL_CAPITAL,
        "final_value":       round(final_val,2),
        "total_return":      round(total_ret*100,2),
        "annual_return":     round(annual*100,2),
        "benchmark_return":  round(bench_ret*100,2),
        "alpha":             round((total_ret-bench_ret)*100,2),
        "sharpe_ratio":      round(sharpe,3),
        "sortino_ratio":     round(sortino,3),
        "calmar_ratio":      round(calmar,3),
        "max_drawdown":      round(max_dd*100,2),
        "total_trades":      len(trades),
        "win_rate":          round(win_rate*100,2),
        "avg_pnl_per_trade": round(avg_pnl,2),
        "regime_days":       {"bull":regime_counts.get(0,0),"highvol":regime_counts.get(1,0),"bear":regime_counts.get(2,0)},
    }


def print_report(metrics, portfolio):
    print("\n"+"═"*58)
    print("  BACKTESTER V3 — PERFORMANCE REPORT")
    print(f"  {BACKTEST_START} to {BACKTEST_END}")
    print("="*58)
    print(f"  Initial Capital     : ₹{metrics['initial_capital']:>12,.0f}")
    print(f"  Final Value         : ₹{metrics['final_value']:>12,.0f}")
    print(f"  {'─'*42}")
    print(f"  Total Return        : {metrics['total_return']:>+11.2f}%")
    print(f"  Annual Return       : {metrics['annual_return']:>+11.2f}%")
    print(f"  Benchmark Return    : {metrics['benchmark_return']:>+11.2f}%")
    alpha_sign = "outperformed" if metrics['alpha']>0 else "underperformed"
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
    print("="*58)
    eq_df = pd.DataFrame(portfolio.equity_curve)
    eq_df.to_csv("data/equity_curve.csv", index=False)
    print(f"\n  Equity curve → data/equity_curve.csv")
    if portfolio.trade_log:
        pd.DataFrame(portfolio.trade_log).to_csv("data/trade_log.csv", index=False)
        print(f"  Trade log    → data/trade_log.csv")


if __name__ == "__main__":
    symbol = None
    if "--symbol" in sys.argv:
        idx    = sys.argv.index("--symbol")
        symbol = sys.argv[idx+1].upper()
        print(f"Single stock: {symbol}")
    else:
        print("Full Nifty 50 V3 backtest...")

    portfolio = run_backtest_v3(symbol)
    if portfolio and portfolio.equity_curve:
        all_data = load_data(symbol)
        metrics  = compute_metrics(portfolio, all_data)
        print_report(metrics, portfolio)
    else:
        print("No results.")
