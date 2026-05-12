"""
central_command_v1.py
---------------------
Central Command V1 — Maximum Returns Strategy.
 
Philosophy:
  - Default: always in V5 passive (max returns)
  - Only switch to V4 active during confirmed bear regime (40+ days)
  - Switch back to V5 immediately when bull resumes
  - Emergency switch to V4 if drawdown hits -15% while in V5
  - Minimize regime switches to reduce transaction cost drag
 
Target: +180-210% return, -20 to -25% max drawdown
 
Usage:
    python central_command_v1.py
"""
 
import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass, field
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk.risk_engine_v4 import RiskConfig
 
DB_PATH         = "data/nifty50.db"
INITIAL_CAPITAL = 100_000.0
BACKTEST_START  = "2018-01-01"
BACKTEST_END    = "2026-04-16"
COMMISSION_PCT  = 0.001
REBALANCE_FREQ  = 5
REGIME_FREQ     = 20
 
# ── CENTRAL COMMAND PARAMETERS ────────────────────────────────────────────────
INDEX_TOP_N          = 48    # all Nifty 50 stocks in passive basket
BEAR_CONFIRM_DAYS    = 40    # days of bear regime before switching to V4
EMERGENCY_DRAWDOWN   = 0.15  # emergency switch to V4 if V5 drops this much
# ──────────────────────────────────────────────────────────────────────────────
 
 
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"""
        SELECT Date, Symbol, Open, High, Low, Close, Volume, Daily_Return
        FROM prices WHERE Date BETWEEN '{BACKTEST_START}' AND '{BACKTEST_END}'
        ORDER BY Symbol, Date
    """, conn)
    conn.close()
    return df
 
 
def get_top_n_symbols(df, n=INDEX_TOP_N):
    avg_vol = df.groupby("Symbol")["Volume"].mean().nlargest(n)
    return list(avg_vol.index)
 
 
# ── SIGNAL FUNCTIONS (V4) ─────────────────────────────────────────────────────
 
def kalman_velocity(close):
    Q, R = 1e-4, 1e-2
    x = np.array([float(close.iloc[0]), 0.0]); P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); H = np.array([[1.0, 0.0]])
    vels = []
    for price in close.values:
        xp = F @ x; Pp = F @ P @ F.T + Q * np.eye(2)
        S  = float((H @ Pp @ H.T)[0, 0]) + R
        K  = (Pp @ H.T) / S; Kf = K.flatten()
        inn = float(price) - float((H @ xp)[0])
        x = xp + Kf * inn; P = (np.eye(2) - Kf[:, None] * H) @ Pp
        vels.append(float(x[1]))
    vel = pd.Series(vels, index=close.index)
    std = vel.rolling(60).std().replace(0, np.nan)
    return (vel / std).clip(-2, 2) / 2
 
 
def adx_fast(high, low, close, period=14):
    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
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
        se  = np.sqrt(res.var() * np.linalg.inv(X.T @ X)[0, 0])
        return float(b[0] / (se + 1e-10))
    except:
        return 0.0
 
 
def hurst_fast(series, lags=20):
    ts = series.dropna().values
    if len(ts) < 40: return 0.5
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in range(2, min(lags, len(ts) // 4))]
    if len(tau) < 5: return 0.5
    try:
        H = np.polyfit(np.log(range(2, len(tau) + 2)), np.log(tau), 1)[0]
        return float(np.clip(H, 0, 1))
    except:
        return 0.5
 
 
def gbm_prob_gain(returns, window=60, horizon=5):
    recent = returns.iloc[-window:].dropna()
    if len(recent) < 20: return 0.5
    mu = float(recent.mean()); sigma = float(recent.std())
    if sigma == 0: return 0.5
    d = (mu - 0.5 * sigma ** 2) * horizon / (sigma * np.sqrt(horizon))
    return float(0.5 * (1 + np.tanh(d * 0.8)))
 
 
def heston_vol_signal(returns, window=60):
    ret = returns.dropna().iloc[-window:]
    if len(ret) < 30: return 0.0, 1.0
    rv    = ret.rolling(21).var().dropna()
    if len(rv) < 10: return 0.0, 1.0
    v0    = float(rv.iloc[-1]); theta = float(rv.mean())
    ratio = v0 / (theta + 1e-8)
    if   ratio > 1.5:  sig = -0.25
    elif ratio > 1.2:  sig = -0.10
    elif ratio < 0.7:  sig = +0.15
    elif ratio < 0.85: sig = +0.07
    else:              sig =  0.0
    return sig, ratio
 
 
def lstm_proxy(close, window=60):
    if len(close) < window: return 0.0
    ret          = close.pct_change().fillna(0).iloc[-window:]
    last_5       = ret.iloc[-5:]; last_20 = ret.iloc[-20:]
    momentum_acc = float(last_5.mean() - last_20.mean())
    streak       = sum(1 if r > 0 else (-1 if r < 0 else 0) for r in ret.iloc[-5:].values)
    return float(np.clip(0.6 * np.clip(momentum_acc * 50, -1, 1) + 0.4 * streak / 5, -1, 1))
 
 
def transformer_proxy(close, window=60):
    if len(close) < window + 20: return 0.0
    returns        = close.pct_change().fillna(0)
    current_window = returns.iloc[-window:].values
    c_mean         = current_window.mean()
    c_std          = current_window.std() + 1e-8
    current_norm   = (current_window - c_mean) / c_std
    similarities   = []; outcomes = []
    for i in range(window, len(returns) - 5, 5):
        hist_window = returns.iloc[i - window:i].values
        h_mean = hist_window.mean(); h_std = hist_window.std() + 1e-8
        hist_norm   = (hist_window - h_mean) / h_std
        sim         = float(np.dot(current_norm, hist_norm) /
                            (np.linalg.norm(current_norm) * np.linalg.norm(hist_norm) + 1e-8))
        outcome     = float(returns.iloc[i:i + 5].mean())
        similarities.append(sim); outcomes.append(outcome)
    if not similarities: return 0.0
    sims             = np.array(similarities)
    attn             = np.exp(sims * 10) / (np.exp(sims * 10).sum() + 1e-8)
    weighted_outcome = float(np.dot(attn, outcomes))
    return float(np.clip(weighted_outcome * 50, -1, 1))
 
 
def rl_dynamic_weights(agent_scores, market_features):
    vol    = market_features.get("vol", 0.5)
    trend  = market_features.get("trend", 0.0)
    regime = market_features.get("regime", 0.0)
    if vol > 0.7:
        return {"kalman":0.20,"mean_rev":0.20,"lstm":0.15,"heston":0.20,"monte_carlo":0.15,"sentiment":0.10}
    elif regime > 0.5:
        return {"kalman":0.15,"mean_rev":0.30,"lstm":0.18,"heston":0.15,"monte_carlo":0.12,"sentiment":0.10}
    elif abs(trend) > 0.5:
        return {"kalman":0.35,"mean_rev":0.12,"lstm":0.20,"heston":0.12,"monte_carlo":0.12,"sentiment":0.09}
    else:
        return {"kalman":0.25,"mean_rev":0.20,"lstm":0.18,"heston":0.12,"monte_carlo":0.13,"sentiment":0.12}
 
 
def detect_regime_fast(market_returns, window=60):
    if len(market_returns) < window: return 0
    recent = market_returns.iloc[-window:].dropna()
    vol    = float(recent.std()); trend = float(recent.mean())
    vol_z  = (vol - market_returns.std()) / (market_returns.std() + 1e-8)
    if   trend < -0.0003: return 2
    elif vol_z > 1.0:     return 1
    else:                 return 0
 
 
def compute_signals_v4(sym_data, current_idx, regime=0):
    hist = sym_data.iloc[:current_idx + 1].copy()
    if len(hist) < 60: return 0.0, 1.0
    close   = hist["Close"]; high = hist["High"]; low = hist["Low"]
    returns = hist["Daily_Return"].fillna(0)
    vel       = kalman_velocity(close)
    adx       = adx_fast(high, low, close)
    mom_score = float((vel * (adx / 25).clip(0.3, 1.0)).clip(-1, 1).iloc[-1])
    if np.isnan(mom_score): mom_score = 0.0
    adf_s      = adf_stat_fast(close.iloc[-120:]) if len(close) >= 120 else 0.0
    H          = hurst_fast(close)
    hurst_mult = max((0.5 - H) * 4, 0) if H < 0.5 else 0.0
    rm = close.rolling(20).mean(); rs = close.rolling(20).std()
    zs         = (close - rm) / rs.replace(0, np.nan)
    adf_gate   = 1.0 if adf_s < -2.0 else 0.3
    mr_score   = float(-(zs / 3).clip(-1, 1).iloc[-1] * adf_gate * (0.5 + 0.5 * hurst_mult))
    if np.isnan(mr_score): mr_score = 0.0
    lstm_score    = lstm_proxy(close)
    transformer_s = transformer_proxy(close)
    heston_s, vol_ratio = heston_vol_signal(returns)
    prob_up       = gbm_prob_gain(returns)
    mc_score      = float((prob_up - 0.5) * 2)
    sent_score    = 0.0
    vol   = float(returns.iloc[-60:].std()) if len(returns) >= 60 else 0.02
    trend = float(returns.iloc[-60:].mean()) if len(returns) >= 60 else 0.0
    market_features = {
        "vol":      np.clip(vol * 50, 0, 1),
        "trend":    np.clip(trend * 100, -1, 1),
        "regime":   regime / 2.0,
        "momentum": 0.0,
    }
    agent_scores = {
        "kalman":      mom_score,  "mean_rev":   mr_score,
        "lstm":        lstm_score, "heston":     heston_s,
        "monte_carlo": mc_score,   "sentiment":  sent_score,
    }
    rl_weights = rl_dynamic_weights(agent_scores, market_features)
    combined = (
        rl_weights["kalman"]      * mom_score   +
        rl_weights["mean_rev"]    * mr_score     +
        rl_weights["lstm"]        * lstm_score   +
        rl_weights["heston"]      * heston_s     +
        rl_weights["monte_carlo"] * mc_score     +
        rl_weights["sentiment"]   * sent_score   +
        0.05                      * transformer_s
    )
    return float(np.clip(combined, -1, 1)), vol_ratio
 
 
# ── PORTFOLIO ─────────────────────────────────────────────────────────────────
 
@dataclass
class CCPortfolio:
    cash:         float = INITIAL_CAPITAL
    positions:    dict  = field(default_factory=dict)
    equity_curve: list  = field(default_factory=list)
    trade_log:    list  = field(default_factory=list)
    peak_value:   float = INITIAL_CAPITAL
    mode:         str   = "V5"   # current active mode: "V5" or "V4"
    mode_log:     list  = field(default_factory=list)
 
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
 
    def sell(self, symbol, price, date, reason="signal"):
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
 
    def liquidate_all(self, day_prices, date, reason="mode_switch"):
        for sym in list(self.positions.keys()):
            if sym in day_prices:
                self.sell(sym, day_prices[sym], date, reason=reason)
 
    def enter_v5(self, top_symbols, day_prices, date):
        """Buy all passive stocks equally."""
        if not top_symbols or not self.cash: return
        per_stock = self.cash / len(top_symbols)
        for sym in top_symbols:
            price = day_prices.get(sym)
            if price and price > 0:
                shares = int(per_stock / (price * (1 + COMMISSION_PCT)))
                if shares > 0:
                    self.buy(sym, shares, price, date)
        self.mode = "V5"
        self.mode_log.append({"date": date, "mode": "V5"})
 
    def enter_v4(self, date):
        """Just switch mode flag — V4 trades via signals."""
        self.mode = "V4"
        self.mode_log.append({"date": date, "mode": "V4"})
 
 
# ── CENTRAL COMMAND ENGINE ────────────────────────────────────────────────────
 
def run_central_command():
    print(f"\nCentral Command V1 ({BACKTEST_START} to {BACKTEST_END})...")
    all_data   = load_data()
    symbols    = all_data["Symbol"].unique()
    all_dates  = sorted(all_data["Date"].unique())
    portfolio  = CCPortfolio()
    config     = RiskConfig(total_capital=INITIAL_CAPITAL)
    market_ret = all_data.groupby("Date")["Daily_Return"].mean()
    top_syms   = get_top_n_symbols(all_data, INDEX_TOP_N)
 
    print(f"Backtesting {len(symbols)} stock(s) across {len(all_dates)} days...")
    print(f"  Default: V5 passive | Bear confirm: {BEAR_CONFIRM_DAYS}d | Emergency DD: {EMERGENCY_DRAWDOWN*100:.0f}%\n")
 
    current_regime     = 0
    bear_days          = 0    # consecutive bear/highvol days
    switches           = 0
 
    # ── Start in V5 passive ────────────────────────────────────────────────
    first_prices = all_data[all_data["Date"] == all_dates[0]].set_index("Symbol")["Close"].to_dict()
    portfolio.enter_v5(top_syms, first_prices, all_dates[0])
 
    for day_idx, date in enumerate(all_dates):
        day_prices = all_data[all_data["Date"] == date].set_index("Symbol")["Close"].to_dict()
 
        # ── Regime detection ───────────────────────────────────────────────
        if day_idx % REGIME_FREQ == 0:
            mr_series      = market_ret[market_ret.index <= date]
            current_regime = detect_regime_fast(mr_series)
 
            if current_regime in [1, 2]:  # bear or high vol
                bear_days += REGIME_FREQ
            else:
                bear_days = 0
 
            total_val = portfolio.market_value(day_prices)
            dd        = portfolio.drawdown(total_val)
 
            # ── Switch V5 → V4 ────────────────────────────────────────────
            if portfolio.mode == "V5":
                if bear_days >= BEAR_CONFIRM_DAYS or dd >= EMERGENCY_DRAWDOWN:
                    reason = "bear_confirmed" if bear_days >= BEAR_CONFIRM_DAYS else "emergency_dd"
                    print(f"  [{date}] Switching V5 → V4 ({reason}, DD={dd*100:.1f}%)")
                    portfolio.liquidate_all(day_prices, date, reason=reason)
                    portfolio.enter_v4(date)
                    switches += 1
 
            # ── Switch V4 → V5 ────────────────────────────────────────────
            elif portfolio.mode == "V4":
                if current_regime == 0 and bear_days == 0:
                    print(f"  [{date}] Switching V4 → V5 (bull resumed)")
                    portfolio.liquidate_all(day_prices, date, reason="bull_resumed")
                    portfolio.enter_v5(top_syms, day_prices, date)
                    switches += 1
 
        # ── V5 MODE: just hold ─────────────────────────────────────────────
        if portfolio.mode == "V5":
            total_value = portfolio.market_value(day_prices)
            portfolio.equity_curve.append({
                "date":            date,
                "portfolio_value": total_value,
                "cash":            portfolio.cash,
                "n_positions":     len(portfolio.positions),
                "regime":          current_regime,
                "mode":            "V5",
            })
            continue
 
        # ── V4 MODE: full active system ────────────────────────────────────
 
        # Stop-loss sweep
        for sym in list(portfolio.positions.keys()):
            if sym not in day_prices: continue
            price = day_prices[sym]
            entry = portfolio.positions[sym]["entry_price"]
            loss  = (price - entry) / entry
            stop  = config.stop_loss_pct * {0: 1.0, 1: 0.85, 2: 0.70}.get(current_regime, 1.0)
            if loss < -stop:
                portfolio.sell(sym, price, date, reason="stop_loss")
 
        if day_idx % REBALANCE_FREQ == 0:
            total_val = portfolio.market_value(day_prices)
            if portfolio.drawdown(total_val) > config.max_drawdown_pct:
                portfolio.liquidate_all(day_prices, date, reason="circuit_breaker")
                portfolio.equity_curve.append({
                    "date":            date,
                    "portfolio_value": portfolio.market_value(day_prices),
                    "cash":            portfolio.cash,
                    "n_positions":     0,
                    "regime":          current_regime,
                    "mode":            "V4",
                })
                continue
 
            for sym in symbols:
                sym_data    = all_data[all_data["Symbol"] == sym].reset_index(drop=True)
                mask        = sym_data["Date"] == date
                if not mask.any(): continue
                current_idx   = sym_data[mask].index[0]
                current_price = day_prices.get(sym)
                if current_price is None or current_price <= 0: continue
 
                result = compute_signals_v4(sym_data, current_idx, current_regime)
                if isinstance(result, tuple): signal, vol_ratio = result
                else:                         signal = result; vol_ratio = 1.0
 
                returns  = sym_data["Daily_Return"].iloc[:current_idx].dropna()
                prob_up  = gbm_prob_gain(returns)
                b        = 0.018 / 0.012
                kelly_f  = max((prob_up * b - (1 - prob_up)) / b, 0)
                heston_pos_mult = 0.70 if vol_ratio > 1.3 else (1.15 if vol_ratio < 0.8 else 1.0)
                regime_s        = {0: 1.0, 1: 0.80, 2: 0.50}.get(current_regime, 1.0)
                size_frac       = min(
                    kelly_f * config.kelly_fraction * regime_s * heston_pos_mult,
                    config.max_position_pct,
                )
 
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
            "date":            date,
            "portfolio_value": total_value,
            "cash":            portfolio.cash,
            "n_positions":     len(portfolio.positions),
            "regime":          current_regime,
            "mode":            "V4",
        })
 
    print(f"\n  Total mode switches: {switches}")
    return portfolio
 
 
# ── METRICS ───────────────────────────────────────────────────────────────────
 
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
    mode_counts  = equity["mode"].value_counts().to_dict() if "mode" in equity.columns else {}
    regime_counts = equity["regime"].value_counts().to_dict() if "regime" in equity.columns else {}
 
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
        "days_in_v4":        mode_counts.get("V4", 0),
        "regime_days": {
            "bull":    regime_counts.get(0, 0),
            "highvol": regime_counts.get(1, 0),
            "bear":    regime_counts.get(2, 0),
        },
    }
 
 
# ── REPORT ────────────────────────────────────────────────────────────────────
 
def print_report(metrics, portfolio):
    print("\n" + "═" * 58)
    print("  CENTRAL COMMAND V1 — MAX RETURNS")
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
    print(f"  CC1 Max Drawdown    : {metrics['max_drawdown']:>+11.2f}%")
    print(f"  Benchmark Max DD    : {metrics['bench_max_dd']:>+11.2f}%")
    print(f"  {'─'*42}")
    print(f"  Total Trades        : {metrics['total_trades']:>12}")
    print(f"  Win Rate            : {metrics['win_rate']:>11.2f}%")
    print(f"  Avg P&L / Trade     : ₹{metrics['avg_pnl_per_trade']:>11,.2f}")
    print(f"  {'─'*42}")
    print(f"  Days in V5 passive  : {metrics['days_in_v5']:>12}")
    print(f"  Days in V4 active   : {metrics['days_in_v4']:>12}")
    rd = metrics["regime_days"]
    print(f"  Days in Bull        : {rd.get('bull',0):>12}")
    print(f"  Days in High Vol    : {rd.get('highvol',0):>12}")
    print(f"  Days in Bear        : {rd.get('bear',0):>12}")
    print("=" * 58)
    print("\n  ── Full Comparison ─────────────────────────────────")
    print(f"  {'Metric':<22} {'V4':>8} {'CC1':>8} {'V6':>8} {'V5':>8} {'Bench':>8}")
    print(f"  {'─'*62}")
    print(f"  {'Total Return':<22} {'+66.34%':>8} {metrics['total_return']:>+7.2f}% {'+110.40%':>8} {'+204.47%':>8} {'+176.48%':>8}")
    print(f"  {'Max Drawdown':<22} {'-16.19%':>8} {metrics['max_drawdown']:>+7.2f}% {'-21.74%':>8} {'-40.63%':>8} {metrics['bench_max_dd']:>+7.2f}%")
    print("=" * 58)
 
    eq_df = pd.DataFrame(portfolio.equity_curve)
    eq_df.to_csv("data/equity_curve_cc1.csv", index=False)
    print(f"\n  Equity curve → data/equity_curve_cc1.csv")
    if portfolio.trade_log:
        pd.DataFrame(portfolio.trade_log).to_csv("data/trade_log_cc1.csv", index=False)
        print(f"  Trade log    → data/trade_log_cc1.csv")
    if portfolio.mode_log:
        pd.DataFrame(portfolio.mode_log).to_csv("data/mode_log_cc1.csv", index=False)
        print(f"  Mode log     → data/mode_log_cc1.csv")
 
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("Central Command V1 — Maximum Returns Strategy")
    portfolio = run_central_command()
    if portfolio and portfolio.equity_curve:
        all_data = load_data()
        metrics  = compute_metrics(portfolio, all_data)
        print_report(metrics, portfolio)
    else:
        print("No results.")
 