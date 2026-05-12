"""
portfolio_allocator.py
----------------------
Portfolio Allocator — Three strategies running simultaneously.

Capital allocation:
  - 40% → V5 Passive (top 100 Nifty 500 by market cap) — max returns
  - 40% → V4 Active (top 48 large caps, full 9-agent system) — crisis protection
  - 20% → CC2 Early Warning (V5 passive + momentum/breadth/vol exit to cash) — risk adjusted

All three run in parallel on the same Nifty 500 database.
Final portfolio value = sum of all three sub-portfolios.

Usage:
    python portfolio_allocator.py
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk.risk_engine_v4 import RiskConfig

DB_PATH         = "data/nifty500.db"
INITIAL_CAPITAL = 100_000.0
BACKTEST_START  = "2018-01-01"
BACKTEST_END    = "2026-04-30"
COMMISSION_PCT  = 0.001
REBALANCE_FREQ  = 5
REGIME_FREQ     = 20

# ── CAPITAL ALLOCATION ────────────────────────────────────────────────────────
V5_FRACTION  = 0.70   # passive top 100 Nifty 500
V4_FRACTION  = 0.10   # active V4 on top 48 large caps
CC2_FRACTION = 0.20   # early warning system
# ──────────────────────────────────────────────────────────────────────────────

# ── CC2 PARAMETERS ────────────────────────────────────────────────────────────
MOMENTUM_WINDOW    = 10
VOL_WINDOW         = 60
VOL_SPIKE_MULT     = 2.0
BREADTH_THRESHOLD  = 0.60
MA_WINDOW          = 20
SIGNALS_TO_EXIT    = 2
COOLDOWN_DAYS      = 20
EMERGENCY_DD_CC2   = 0.20
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


def get_top_n_by_market_cap(df, n):
    df2 = df.copy()
    df2["mktcap_proxy"] = df2["Close"] * df2["Volume"]
    return list(df2.groupby("Symbol")["mktcap_proxy"].mean().nlargest(n).index)


def get_top_n_by_volume(df, n):
    return list(df.groupby("Symbol")["Volume"].mean().nlargest(n).index)


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
    return float(np.clip(float(np.dot(attn, outcomes)) * 50, -1, 1))


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
    vol   = float(returns.iloc[-60:].std()) if len(returns) >= 60 else 0.02
    trend = float(returns.iloc[-60:].mean()) if len(returns) >= 60 else 0.0
    mf = {"vol": np.clip(vol*50,0,1), "trend": np.clip(trend*100,-1,1), "regime": regime/2.0, "momentum": 0.0}
    aw = {"kalman": mom_score, "mean_rev": mr_score, "lstm": lstm_score, "heston": heston_s, "monte_carlo": mc_score, "sentiment": 0.0}
    rw = rl_dynamic_weights(aw, mf)
    combined = (rw["kalman"]*mom_score + rw["mean_rev"]*mr_score + rw["lstm"]*lstm_score +
                rw["heston"]*heston_s + rw["monte_carlo"]*mc_score + 0.05*transformer_s)
    return float(np.clip(combined, -1, 1)), vol_ratio


# ── CC2 EARLY WARNING ─────────────────────────────────────────────────────────

def compute_early_warning(all_data, top_syms, date):
    hist       = all_data[all_data["Date"] <= date]
    market_ret = hist.groupby("Date")["Daily_Return"].mean().dropna()
    if len(market_ret) < VOL_WINDOW + MOMENTUM_WINDOW:
        return 0, {}
    signals = {}
    recent_10d = market_ret.iloc[-MOMENTUM_WINDOW:].sum()
    prior_10d  = market_ret.iloc[-MOMENTUM_WINDOW*2:-MOMENTUM_WINDOW].sum()
    signals["momentum"] = (recent_10d < 0) and (prior_10d > 0)
    baseline_vol = market_ret.iloc[-VOL_WINDOW:-MOMENTUM_WINDOW].std()
    current_vol  = market_ret.iloc[-MOMENTUM_WINDOW:].std()
    signals["volatility"] = (current_vol > VOL_SPIKE_MULT * baseline_vol) if baseline_vol > 0 else False
    below_ma = 0; total = 0
    for sym in top_syms:
        sym_hist = hist[hist["Symbol"] == sym]["Close"]
        if len(sym_hist) < MA_WINDOW: continue
        total += 1
        if sym_hist.iloc[-1] < sym_hist.iloc[-MA_WINDOW:].mean():
            below_ma += 1
    signals["breadth"] = (below_ma / total > BREADTH_THRESHOLD) if total > 0 else False
    return sum(signals.values()), signals


# ── SUB PORTFOLIOS ────────────────────────────────────────────────────────────

@dataclass
class SubPortfolio:
    name:         str
    cash:         float
    positions:    dict  = field(default_factory=dict)
    trade_log:    list  = field(default_factory=list)
    peak_value:   float = 0.0
    mode:         str   = "active"
    days_in_cash: int   = 0

    def __post_init__(self):
        self.peak_value = self.cash

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
        self.trade_log.append({"date": date, "symbol": symbol, "action": "BUY",
                               "shares": shares, "price": price, "pnl": None, "strategy": self.name})
        return True

    def sell(self, symbol, price, date, reason="signal"):
        if symbol not in self.positions: return False
        shares = self.positions[symbol]["shares"]
        entry  = self.positions[symbol]["entry_price"]
        pnl    = (price - entry) * shares
        self.cash += shares * price * (1 - COMMISSION_PCT)
        del self.positions[symbol]
        self.trade_log.append({"date": date, "symbol": symbol, "action": "SELL",
                               "shares": shares, "price": price, "pnl": pnl,
                               "reason": reason, "strategy": self.name})
        return True

    def liquidate_all(self, day_prices, date, reason="exit"):
        for sym in list(self.positions.keys()):
            if sym in day_prices:
                self.sell(sym, day_prices[sym], date, reason=reason)

    def buy_basket(self, symbols, day_prices, date):
        if not symbols or self.cash <= 0: return
        per_stock = self.cash / len(symbols)
        for sym in symbols:
            price = day_prices.get(sym)
            if price and price > 0:
                shares = int(per_stock / (price * (1 + COMMISSION_PCT)))
                if shares > 0:
                    self.buy(sym, shares, price, date)


# ── MAIN ALLOCATOR ────────────────────────────────────────────────────────────

def run_portfolio_allocator():
    print(f"\nPortfolio Allocator ({BACKTEST_START} to {BACKTEST_END})")
    print(f"  V5 Passive: {V5_FRACTION*100:.0f}% | V4 Active: {V4_FRACTION*100:.0f}% | CC2 Early Warning: {CC2_FRACTION*100:.0f}%\n")

    all_data   = load_data()
    all_dates  = sorted(all_data["Date"].unique())
    market_ret = all_data.groupby("Date")["Daily_Return"].mean()

    # Stock universes
    v5_symbols  = get_top_n_by_market_cap(all_data, 100)  # top 100 by mktcap
    v4_symbols  = get_top_n_by_market_cap(all_data, 48)   # top 48 large caps
    cc2_symbols = get_top_n_by_market_cap(all_data, 100)  # same as V5

    print(f"  V5 universe : top 100 by market cap")
    print(f"  V4 universe : top 48 by market cap")
    print(f"  CC2 universe: top 100 by market cap")
    print(f"  Total stocks: {len(all_data['Symbol'].unique())}\n")

    # Initialize sub portfolios
    v5  = SubPortfolio("V5_Passive",  INITIAL_CAPITAL * V5_FRACTION,  mode="passive")
    v4  = SubPortfolio("V4_Active",   INITIAL_CAPITAL * V4_FRACTION,  mode="active")
    cc2 = SubPortfolio("CC2_Warning", INITIAL_CAPITAL * CC2_FRACTION, mode="V5")

    config = RiskConfig(total_capital=INITIAL_CAPITAL * V4_FRACTION)

    # V5: buy and hold from day 1
    first_day    = all_dates[0]
    first_prices = all_data[all_data["Date"] == first_day].set_index("Symbol")["Close"].to_dict()
    v5.buy_basket(v5_symbols, first_prices, first_day)

    # CC2: start in V5 mode
    cc2.buy_basket(cc2_symbols, first_prices, first_day)

    # Track equity
    equity_curve   = []
    current_regime = 0

    for day_idx, date in enumerate(all_dates):
        day_prices = all_data[all_data["Date"] == date].set_index("Symbol")["Close"].to_dict()

        # Regime detection
        if day_idx % REGIME_FREQ == 0:
            mr_series      = market_ret[market_ret.index <= date]
            current_regime = detect_regime_fast(mr_series)

        # ── CC2: early warning check ───────────────────────────────────────
        if cc2.mode == "CASH":
            cc2.days_in_cash += 1

        if day_idx % 5 == 0 and day_idx > VOL_WINDOW + MOMENTUM_WINDOW:
            n_sig, sigs = compute_early_warning(all_data, cc2_symbols, date)

            if cc2.mode == "V5" and n_sig >= SIGNALS_TO_EXIT:
                cc2.liquidate_all(day_prices, date, reason="early_warning")
                cc2.mode = "CASH"
                cc2.days_in_cash = 0

            elif cc2.mode == "CASH" and n_sig == 0 and cc2.days_in_cash >= COOLDOWN_DAYS:
                cc2.buy_basket(cc2_symbols, day_prices, date)
                cc2.mode = "V5"
                cc2.days_in_cash = 0

        # CC2 emergency exit
        cc2_val = cc2.market_value(day_prices)
        if cc2.mode == "V5" and cc2.drawdown(cc2_val) >= EMERGENCY_DD_CC2:
            cc2.liquidate_all(day_prices, date, reason="emergency_dd")
            cc2.mode = "CASH"
            cc2.days_in_cash = 0

        # ── V4: full active system ─────────────────────────────────────────
        if day_idx % REBALANCE_FREQ == 0:
            v4_val = v4.market_value(day_prices)

            if v4.drawdown(v4_val) > config.max_drawdown_pct:
                v4.liquidate_all(day_prices, date, reason="circuit_breaker")
            else:
                # Stop losses
                for sym in list(v4.positions.keys()):
                    if sym not in day_prices: continue
                    price = day_prices[sym]
                    entry = v4.positions[sym]["entry_price"]
                    loss  = (price - entry) / entry
                    stop  = config.stop_loss_pct * {0:1.0,1:0.85,2:0.70}.get(current_regime,1.0)
                    if loss < -stop:
                        v4.sell(sym, price, date, reason="stop_loss")

                # Signals
                for sym in v4_symbols:
                    sym_data    = all_data[all_data["Symbol"] == sym].reset_index(drop=True)
                    mask        = sym_data["Date"] == date
                    if not mask.any(): continue
                    current_idx   = sym_data[mask].index[0]
                    current_price = day_prices.get(sym)
                    if current_price is None or current_price <= 0: continue

                    result = compute_signals_v4(sym_data, current_idx, current_regime)
                    signal, vol_ratio = result if isinstance(result, tuple) else (result, 1.0)

                    returns  = sym_data["Daily_Return"].iloc[:current_idx].dropna()
                    prob_up  = gbm_prob_gain(returns)
                    b        = 0.018 / 0.012
                    kelly_f  = max((prob_up * b - (1 - prob_up)) / b, 0)
                    heston_m = 0.70 if vol_ratio > 1.3 else (1.15 if vol_ratio < 0.8 else 1.0)
                    regime_s = {0:1.0,1:0.80,2:0.50}.get(current_regime,1.0)
                    size_frac = min(kelly_f * config.kelly_fraction * regime_s * heston_m,
                                   config.max_position_pct)

                    if signal > config.min_signal_strength:
                        if sym not in v4.positions:
                            shares = int(v4.cash * size_frac / current_price)
                            if shares > 0:
                                v4.buy(sym, shares, current_price, date)
                    elif signal < -config.min_signal_strength:
                        if sym in v4.positions:
                            v4.sell(sym, current_price, date, reason="signal")

        # Total portfolio value
        v5_val    = v5.market_value(day_prices)
        v4_val    = v4.market_value(day_prices)
        cc2_val   = cc2.market_value(day_prices)
        total_val = v5_val + v4_val + cc2_val

        equity_curve.append({
            "date":        date,
            "total":       total_val,
            "v5_value":    v5_val,
            "v4_value":    v4_val,
            "cc2_value":   cc2_val,
            "cc2_mode":    cc2.mode,
            "regime":      current_regime,
        })

    return equity_curve, v5, v4, cc2


# ── METRICS + REPORT ──────────────────────────────────────────────────────────

def compute_and_print(equity_curve, v5, v4, cc2, all_data):
    equity       = pd.DataFrame(equity_curve).set_index("date")
    equity.index = pd.to_datetime(equity.index)

    def metrics(series, capital):
        total_ret = (series.iloc[-1] - capital) / capital
        dr        = series.pct_change().dropna()
        ex        = dr - 0.06/252
        sh        = (ex.mean()/ex.std())*np.sqrt(252) if ex.std()>0 else 0
        dn        = ex[ex<0]
        so        = (ex.mean()/dn.std())*np.sqrt(252) if dn.std()>0 else 0
        rm        = series.cummax()
        mdd       = float(((series-rm)/rm).min())
        yrs       = (series.index[-1]-series.index[0]).days/365
        ann       = (1+total_ret)**(1/max(yrs,1))-1
        cal       = ann/abs(mdd) if mdd!=0 else 0
        return {"ret":round(total_ret*100,2),"ann":round(ann*100,2),
                "sh":round(sh,3),"so":round(so,3),"mdd":round(mdd*100,2),"cal":round(cal,3)}

    bench     = all_data.groupby("Date")["Close"].mean()
    bench_ret = (bench.iloc[-1]-bench.iloc[0])/bench.iloc[0]
    bench_rm  = bench.cummax()
    bench_mdd = float(((bench-bench_rm)/bench_rm).min())

    tm = metrics(equity["total"],   INITIAL_CAPITAL)
    v5m = metrics(equity["v5_value"], INITIAL_CAPITAL * V5_FRACTION)
    v4m = metrics(equity["v4_value"], INITIAL_CAPITAL * V4_FRACTION)
    c2m = metrics(equity["cc2_value"],INITIAL_CAPITAL * CC2_FRACTION)

    all_trades = pd.DataFrame(v5.trade_log + v4.trade_log + cc2.trade_log)
    total_trades = len(all_trades)

    print("\n" + "═"*62)
    print("  PORTFOLIO ALLOCATOR — COMBINED RESULTS")
    print(f"  {BACKTEST_START} to {BACKTEST_END}")
    print(f"  V5 {V5_FRACTION*100:.0f}% | V4 {V4_FRACTION*100:.0f}% | CC2 {CC2_FRACTION*100:.0f}%")
    print("="*62)
    print(f"  Initial Capital     : ₹{INITIAL_CAPITAL:>12,.0f}")
    print(f"  Final Value         : ₹{equity['total'].iloc[-1]:>12,.0f}")
    print(f"  {'─'*46}")
    print(f"  Total Return        : {tm['ret']:>+11.2f}%")
    print(f"  Annual Return       : {tm['ann']:>+11.2f}%")
    print(f"  Benchmark Return    : {bench_ret*100:>+11.2f}%")
    print(f"  Alpha               : {tm['ret']-bench_ret*100:>+11.2f}%")
    print(f"  {'─'*46}")
    print(f"  Sharpe Ratio        : {tm['sh']:>12.3f}")
    print(f"  Sortino Ratio       : {tm['so']:>12.3f}")
    print(f"  Calmar Ratio        : {tm['cal']:>12.3f}")
    print(f"  Max Drawdown        : {tm['mdd']:>+11.2f}%")
    print(f"  Benchmark Max DD    : {bench_mdd*100:>+11.2f}%")
    print(f"  {'─'*46}")
    print(f"  Total Trades        : {total_trades:>12}")
    print("="*62)
    print("\n  ── Sub-Portfolio Breakdown ──────────────────────────────")
    print(f"  {'Strategy':<20} {'Alloc':>6} {'Return':>8} {'Max DD':>8} {'Sharpe':>8} {'Calmar':>8}")
    print(f"  {'─'*62}")
    print(f"  {'V5 Passive (100)':<20} {'40%':>6} {v5m['ret']:>+7.2f}% {v5m['mdd']:>+7.2f}% {v5m['sh']:>8.3f} {v5m['cal']:>8.3f}")
    print(f"  {'V4 Active (48)':<20} {'40%':>6} {v4m['ret']:>+7.2f}% {v4m['mdd']:>+7.2f}% {v4m['sh']:>8.3f} {v4m['cal']:>8.3f}")
    print(f"  {'CC2 Early Warn':<20} {'20%':>6} {c2m['ret']:>+7.2f}% {c2m['mdd']:>+7.2f}% {c2m['sh']:>8.3f} {c2m['cal']:>8.3f}")
    print(f"  {'─'*62}")
    print(f"  {'COMBINED':<20} {'100%':>6} {tm['ret']:>+7.2f}% {tm['mdd']:>+7.2f}% {tm['sh']:>8.3f} {tm['cal']:>8.3f}")
    print("="*62)

    equity.to_csv("data/equity_curve_allocator.csv")
    all_trades.to_csv("data/trade_log_allocator.csv", index=False)
    print(f"\n  Equity curve → data/equity_curve_allocator.csv")
    print(f"  Trade log    → data/trade_log_allocator.csv")


if __name__ == "__main__":
    print("Portfolio Allocator — Three Strategies Running Simultaneously")
    equity_curve, v5, v4, cc2 = run_portfolio_allocator()
    all_data = load_data()
    compute_and_print(equity_curve, v5, v4, cc2, all_data)
