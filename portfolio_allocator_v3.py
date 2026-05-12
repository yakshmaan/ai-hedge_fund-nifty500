"""
portfolio_allocator_v3.py
-------------------------
Portfolio Allocator V3 — Dual Database + Flexi-SIP.

Capital allocation:
  - 80% → V5 Passive top 100 Nifty 500 (nifty500.db)
  - 20% → V4 Active Nifty 50 (nifty50.db)

Flexi-SIP:
  - Bull regime (HMM=0) → add ₹10,000/month (₹8,000 V5 + ₹2,000 V4)
  - Bear/High-vol regime → add ₹5,000/month (₹4,000 V5 + ₹1,000 V4)
  - Monthly additions buy into existing positions proportionally

Usage:
    python portfolio_allocator_v3.py
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk.risk_engine_v4 import RiskConfig

# ── TWO DATABASES ─────────────────────────────────────────────────────────────
DB_PATH_50  = "data/nifty50.db"
DB_PATH_500 = "data/nifty500.db"
# ──────────────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 100_000.0
BACKTEST_START  = "2018-01-01"
BACKTEST_END    = "2026-04-16"
COMMISSION_PCT  = 0.001
REBALANCE_FREQ  = 5
REGIME_FREQ     = 20

# ── CAPITAL ALLOCATION ────────────────────────────────────────────────────────
V5_FRACTION = 0.80
V4_FRACTION = 0.20
# ──────────────────────────────────────────────────────────────────────────────

# ── FLEXI-SIP PARAMETERS ──────────────────────────────────────────────────────
SIP_BULL_AMOUNT    = 5_000.0   # monthly addition in bull regime
SIP_BEAR_AMOUNT    = 5_000.0   # monthly addition in bear/high-vol regime
SIP_DAY_OF_MONTH   = 1          # add on first trading day of each month
# ──────────────────────────────────────────────────────────────────────────────


def load_nifty50():
    conn = sqlite3.connect(DB_PATH_50)
    df   = pd.read_sql(f"""
        SELECT Date, Symbol, Open, High, Low, Close, Volume, Daily_Return
        FROM prices WHERE Date BETWEEN '{BACKTEST_START}' AND '{BACKTEST_END}'
        ORDER BY Symbol, Date
    """, conn)
    conn.close()
    return df


def load_nifty500():
    conn = sqlite3.connect(DB_PATH_500)
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


# ── SIGNAL FUNCTIONS ──────────────────────────────────────────────────────────

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
    sims = np.array(similarities)
    attn = np.exp(sims * 10) / (np.exp(sims * 10).sum() + 1e-8)
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
    aw = {"kalman": mom_score, "mean_rev": mr_score, "lstm": lstm_score,
          "heston": heston_s, "monte_carlo": mc_score, "sentiment": 0.0}
    rw = rl_dynamic_weights(aw, mf)
    combined = (rw["kalman"]*mom_score + rw["mean_rev"]*mr_score + rw["lstm"]*lstm_score +
                rw["heston"]*heston_s + rw["monte_carlo"]*mc_score + 0.05*transformer_s)
    return float(np.clip(combined, -1, 1)), vol_ratio


# ── SUB PORTFOLIO ─────────────────────────────────────────────────────────────

@dataclass
class SubPortfolio:
    name:       str
    cash:       float
    positions:  dict = field(default_factory=dict)
    trade_log:  list = field(default_factory=list)
    peak_value: float = 0.0
    total_invested: float = 0.0  # tracks total capital added including SIP

    def __post_init__(self):
        self.peak_value    = self.cash
        self.total_invested = self.cash

    def market_value(self, prices):
        return self.cash + sum(
            self.positions[s]["shares"] * prices.get(s, 0)
            for s in self.positions
        )

    def drawdown(self, val):
        if val > self.peak_value: self.peak_value = val
        return (self.peak_value - val) / self.peak_value

    def add_sip(self, amount):
        """Add monthly SIP amount to cash."""
        self.cash           += amount
        self.total_invested += amount
        if self.cash > self.peak_value:
            self.peak_value = self.cash

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
                               "shares": shares, "price": price, "pnl": None,
                               "strategy": self.name})
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

    def buy_basket_proportional(self, symbols, day_prices, date):
        """Buy basket proportionally — used for SIP additions."""
        if not symbols or self.cash <= 0: return
        # Buy each stock equally with available cash
        per_stock = self.cash / len(symbols)
        for sym in symbols:
            price = day_prices.get(sym)
            if price and price > 0:
                shares = int(per_stock / (price * (1 + COMMISSION_PCT)))
                if shares > 0:
                    self.buy(sym, shares, price, date)

    def buy_basket_initial(self, symbols, day_prices, date):
        """Buy basket equally — used for initial investment."""
        self.buy_basket_proportional(symbols, day_prices, date)


# ── MAIN ALLOCATOR ────────────────────────────────────────────────────────────

def run_allocator():
    print(f"\nPortfolio Allocator V3 — Dual DB + Flexi-SIP")
    print(f"  {BACKTEST_START} to {BACKTEST_END}")
    print(f"  Initial: ₹{INITIAL_CAPITAL:,.0f} | V5 {V5_FRACTION*100:.0f}% N500 | V4 {V4_FRACTION*100:.0f}% N50")
    print(f"  SIP: ₹{SIP_BULL_AMOUNT:,.0f}/month bull | ₹{SIP_BEAR_AMOUNT:,.0f}/month bear\n")

    # Load both databases
    print("  Loading Nifty 50 data...")
    data50  = load_nifty50()
    print(f"  Nifty 50: {data50['Symbol'].nunique()} stocks")

    print("  Loading Nifty 500 data...")
    data500 = load_nifty500()
    print(f"  Nifty 500: {data500['Symbol'].nunique()} stocks\n")

    all_dates = sorted(set(data50["Date"].unique()) & set(data500["Date"].unique()))
    print(f"  Common trading days: {len(all_dates)}\n")

    # Stock universes
    v4_symbols = data50["Symbol"].unique().tolist()
    v5_symbols = get_top_n_by_market_cap(data500, 100)

    print(f"  V4: {len(v4_symbols)} Nifty 50 stocks")
    print(f"  V5: top 100 Nifty 500 by market cap\n")

    market_ret_50 = data50.groupby("Date")["Daily_Return"].mean()

    # Init sub portfolios
    v5 = SubPortfolio("V5_Passive", INITIAL_CAPITAL * V5_FRACTION)
    v4 = SubPortfolio("V4_Active",  INITIAL_CAPITAL * V4_FRACTION)

    config = RiskConfig(total_capital=INITIAL_CAPITAL * V4_FRACTION)

    # Day 1 — buy passive baskets
    first_day      = all_dates[0]
    prices500_day1 = data500[data500["Date"] == first_day].set_index("Symbol")["Close"].to_dict()
    prices50_day1  = data50[data50["Date"] == first_day].set_index("Symbol")["Close"].to_dict()

    v5.buy_basket_initial(v5_symbols, prices500_day1, first_day)

    equity_curve   = []
    sip_log        = []
    current_regime = 0
    last_sip_month = None
    total_sip_added = 0.0

    for day_idx, date in enumerate(all_dates):
        prices50  = data50[data50["Date"] == date].set_index("Symbol")["Close"].to_dict()
        prices500 = data500[data500["Date"] == date].set_index("Symbol")["Close"].to_dict()

        # Regime detection
        if day_idx % REGIME_FREQ == 0:
            mr50           = market_ret_50[market_ret_50.index <= date]
            current_regime = detect_regime_fast(mr50)

        # ── FLEXI-SIP: add monthly on first trading day of each month ──────
        current_month = date[:7]  # YYYY-MM
        if current_month != last_sip_month:
            last_sip_month = current_month

            # Determine SIP amount based on regime
            sip_total  = SIP_BULL_AMOUNT if current_regime == 0 else SIP_BEAR_AMOUNT
            sip_v5     = sip_total * V5_FRACTION   # 80%
            sip_v4     = sip_total * V4_FRACTION   # 20%

            # Add cash to each sub portfolio
            v5.add_sip(sip_v5)
            v4.add_sip(sip_v4)

            # V5: immediately buy more of passive basket with new cash
            v5.buy_basket_proportional(v5_symbols, prices500, date)

            total_sip_added += sip_total
            sip_log.append({
                "date":    date,
                "regime":  current_regime,
                "amount":  sip_total,
                "v5_sip":  sip_v5,
                "v4_sip":  sip_v4,
                "total_added_so_far": total_sip_added + INITIAL_CAPITAL,
            })

        # ── V4: stop loss sweep ────────────────────────────────────────────
        for sym in list(v4.positions.keys()):
            if sym not in prices50: continue
            price = prices50[sym]
            entry = v4.positions[sym]["entry_price"]
            loss  = (price - entry) / entry
            stop  = config.stop_loss_pct * {0:1.0,1:0.85,2:0.70}.get(current_regime,1.0)
            if loss < -stop:
                v4.sell(sym, price, date, reason="stop_loss")

        # ── V4: rebalance ──────────────────────────────────────────────────
        if day_idx % REBALANCE_FREQ == 0:
            v4_val = v4.market_value(prices50)

            if v4.drawdown(v4_val) > config.max_drawdown_pct:
                v4.liquidate_all(prices50, date, reason="circuit_breaker")
            else:
                for sym in v4_symbols:
                    sym_data    = data50[data50["Symbol"] == sym].reset_index(drop=True)
                    mask        = sym_data["Date"] == date
                    if not mask.any(): continue
                    current_idx   = sym_data[mask].index[0]
                    current_price = prices50.get(sym)
                    if current_price is None or current_price <= 0: continue

                    result = compute_signals_v4(sym_data, current_idx, current_regime)
                    signal, vol_ratio = result if isinstance(result, tuple) else (result, 1.0)

                    returns   = sym_data["Daily_Return"].iloc[:current_idx].dropna()
                    prob_up   = gbm_prob_gain(returns)
                    b         = 0.018 / 0.012
                    kelly_f   = max((prob_up * b - (1 - prob_up)) / b, 0)
                    heston_m  = 0.70 if vol_ratio > 1.3 else (1.15 if vol_ratio < 0.8 else 1.0)
                    regime_s  = {0:1.0,1:0.80,2:0.50}.get(current_regime,1.0)
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

        v5_val = v5.market_value(prices500)
        v4_val = v4.market_value(prices50)

        equity_curve.append({
            "date":          date,
            "total":         v5_val + v4_val,
            "v5_value":      v5_val,
            "v4_value":      v4_val,
            "total_invested": v5.total_invested + v4.total_invested,
            "regime":        current_regime,
        })

    print(f"\n  Total SIP added: ₹{total_sip_added:,.0f}")
    print(f"  Total invested : ₹{INITIAL_CAPITAL + total_sip_added:,.0f}")
    return equity_curve, v5, v4, sip_log, data50, data500


# ── METRICS + REPORT ──────────────────────────────────────────────────────────

def compute_and_print(equity_curve, v5, v4, sip_log, data50, data500):
    equity       = pd.DataFrame(equity_curve).set_index("date")
    equity.index = pd.to_datetime(equity.index)

    total_invested = equity["total_invested"].iloc[-1]
    final_val      = equity["total"].iloc[-1]
    total_profit   = final_val - total_invested
    total_ret      = (final_val - total_invested) / total_invested

    daily_ret = equity["total"].pct_change().dropna()
    rf_daily  = 0.06 / 252
    excess    = daily_ret - rf_daily
    sharpe    = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
    downside  = excess[excess < 0]
    sortino   = (excess.mean() / downside.std()) * np.sqrt(252) if downside.std() > 0 else 0
    roll_max  = equity["total"].cummax()
    max_dd    = float(((equity["total"] - roll_max) / roll_max).min())
    years     = (equity.index[-1] - equity.index[0]).days / 365
    annual    = (1 + total_ret) ** (1 / max(years, 1)) - 1
    calmar    = annual / abs(max_dd) if max_dd != 0 else 0

    bench50   = data50.groupby("Date")["Close"].mean()
    b50_ret   = (bench50.iloc[-1] - bench50.iloc[0]) / bench50.iloc[0]
    b50_rm    = bench50.cummax()
    b50_mdd   = float(((bench50 - b50_rm) / b50_rm).min())

    # SIP breakdown
    sip_df       = pd.DataFrame(sip_log)
    bull_months  = len(sip_df[sip_df["regime"] == 0])
    bear_months  = len(sip_df[sip_df["regime"] != 0])
    bull_sip     = sip_df[sip_df["regime"] == 0]["amount"].sum()
    bear_sip     = sip_df[sip_df["regime"] != 0]["amount"].sum()

    # Sub portfolio metrics
    def sub_metrics(series, invested):
        tr  = (series.iloc[-1] - invested) / invested
        rm  = series.cummax()
        mdd = float(((series - rm) / rm).min())
        dr  = series.pct_change().dropna()
        ex  = dr - 0.06/252
        sh  = (ex.mean()/ex.std())*np.sqrt(252) if ex.std()>0 else 0
        return {"ret": round(tr*100,2), "mdd": round(mdd*100,2), "sh": round(sh,3)}

    v5m = sub_metrics(equity["v5_value"], v5.total_invested)
    v4m = sub_metrics(equity["v4_value"], v4.total_invested)

    print("\n" + "═"*62)
    print("  PORTFOLIO ALLOCATOR V3 — DUAL DB + FLEXI-SIP")
    print(f"  {BACKTEST_START} to {BACKTEST_END}")
    print(f"  V5 {V5_FRACTION*100:.0f}% Nifty500 | V4 {V4_FRACTION*100:.0f}% Nifty50")
    print("="*62)
    print(f"  Initial Capital     : ₹{INITIAL_CAPITAL:>12,.0f}")
    print(f"  Total SIP Added     : ₹{total_invested - INITIAL_CAPITAL:>12,.0f}")
    print(f"  Total Invested      : ₹{total_invested:>12,.0f}")
    print(f"  Final Value         : ₹{final_val:>12,.0f}")
    print(f"  Total Profit        : ₹{total_profit:>12,.0f}")
    print(f"  {'─'*46}")
    print(f"  Return on Invested  : {total_ret*100:>+11.2f}%")
    print(f"  Annual Return       : {annual*100:>+11.2f}%")
    print(f"  Nifty 50 Benchmark  : {b50_ret*100:>+11.2f}%")
    print(f"  {'─'*46}")
    print(f"  Sharpe Ratio        : {sharpe:>12.3f}")
    print(f"  Sortino Ratio       : {sortino:>12.3f}")
    print(f"  Calmar Ratio        : {calmar:>12.3f}")
    print(f"  Max Drawdown        : {max_dd*100:>+11.2f}%")
    print(f"  Benchmark Max DD    : {b50_mdd*100:>+11.2f}%")
    print(f"  {'─'*46}")
    print(f"  Bull months (₹10k)  : {bull_months:>12} → ₹{bull_sip:>10,.0f}")
    print(f"  Bear months (₹5k)   : {bear_months:>12} → ₹{bear_sip:>10,.0f}")
    print(f"  Total months        : {bull_months+bear_months:>12}")
    print(f"  {'─'*46}")
    print(f"  Total Trades        : {len(v5.trade_log)+len(v4.trade_log):>12}")
    print("="*62)
    print("\n  ── Sub-Portfolio Breakdown ──────────────────────────────")
    print(f"  {'Strategy':<22} {'Alloc':>5} {'DB':>5} {'Return':>8} {'Max DD':>8} {'Sharpe':>8}")
    print(f"  {'─'*58}")
    print(f"  {'V5 Passive top 100':<22} {'80%':>5} {'N500':>5} {v5m['ret']:>+7.2f}% {v5m['mdd']:>+7.2f}% {v5m['sh']:>8.3f}")
    print(f"  {'V4 Active Nifty 50':<22} {'20%':>5} {'N50':>5} {v4m['ret']:>+7.2f}% {v4m['mdd']:>+7.2f}% {v4m['sh']:>8.3f}")
    print(f"  {'─'*58}")
    print(f"  {'COMBINED':<22} {'100%':>5} {'BOTH':>5} {total_ret*100:>+7.2f}% {max_dd*100:>+7.2f}% {sharpe:>8.3f}")
    print("="*62)

    # Save outputs
    equity.to_csv("data/equity_curve_allocator_v3.csv")
    pd.DataFrame(v5.trade_log + v4.trade_log).to_csv("data/trade_log_allocator_v3.csv", index=False)
    pd.DataFrame(sip_log).to_csv("data/sip_log_v3.csv", index=False)
    print(f"\n  Equity curve → data/equity_curve_allocator_v3.csv")
    print(f"  Trade log    → data/trade_log_allocator_v3.csv")
    print(f"  SIP log      → data/sip_log_v3.csv")


if __name__ == "__main__":
    print("Portfolio Allocator V3 — Dual Database + Flexi-SIP")
    equity_curve, v5, v4, sip_log, data50, data500 = run_allocator()
    compute_and_print(equity_curve, v5, v4, sip_log, data50, data500)
