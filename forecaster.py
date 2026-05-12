"""
forecaster.py
-------------
Live price forecaster using GBM + Kalman Filter.
 
Features:
  - Fetches latest real price from yfinance
  - Runs 1000 GBM simulations for next 30 days
  - Shows confidence bands (5th, 25th, 75th, 95th percentile)
  - Kalman Filter smoothed trend line
  - Price target (median forecast)
  - Bull/Bear/Base case scenarios
  - Support and resistance levels
 
Used by dashboard.py — import and call run_forecast(symbol)
"""
 
import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings("ignore")
 
DB_PATH       = "data/nifty50.db"
N_SIMS        = 1000
HORIZON_DAYS  = 30
 
 
def get_historical_data(symbol, days=252):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"""
        SELECT Date, Close, High, Low, Volume, Daily_Return
        FROM prices WHERE Symbol='{symbol}'
        ORDER BY Date DESC LIMIT {days}
    """, conn)
    conn.close()
    return df.sort_values("Date").reset_index(drop=True)
 
 
def estimate_params(returns, window=60):
    """Estimate GBM drift and volatility from recent returns."""
    recent = returns.dropna().iloc[-window:]
    if len(recent) < 20:
        return 0.0, 0.02
    log_ret = np.log(1 + recent.replace(-1, np.nan).dropna())
    mu      = float(log_ret.mean())
    sigma   = float(log_ret.std())
    return mu, max(sigma, 1e-6)
 
 
def run_gbm_simulations(S0, mu, sigma, T=HORIZON_DAYS, n_sims=N_SIMS):
    """
    Run n_sims GBM paths over T days.
    Returns array of shape (n_sims, T+1) — includes starting price.
    """
    rng   = np.random.default_rng(42)
    dt    = 1.0
    paths = np.zeros((n_sims, T + 1))
    paths[:, 0] = S0
 
    for t in range(1, T + 1):
        Z = rng.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
 
    return paths
 
 
def kalman_smooth(prices):
    """Apply Kalman filter to smooth price series."""
    Q, R = 1e-4, 1e-2
    x    = np.array([float(prices.iloc[0]), 0.0])
    P    = np.eye(2)
    F    = np.array([[1.0, 1.0], [0.0, 1.0]])
    H    = np.array([[1.0, 0.0]])
    smoothed = []
 
    for price in prices.values:
        xp  = F @ x
        Pp  = F @ P @ F.T + Q * np.eye(2)
        S   = float((H @ Pp @ H.T)[0, 0]) + R
        K   = (Pp @ H.T) / S
        Kf  = K.flatten()
        inn = float(price) - float((H @ xp)[0])
        x   = xp + Kf * inn
        P   = (np.eye(2) - Kf[:, None] * H) @ Pp
        smoothed.append(float(x[0]))
 
    return smoothed
 
 
def support_resistance(df, window=20):
    """Find recent support and resistance levels."""
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
 
    resistance = float(high.rolling(window).max().iloc[-1])
    support    = float(low.rolling(window).min().iloc[-1])
    pivot      = float((high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3)
 
    return {
        "resistance": round(resistance, 2),
        "support":    round(support, 2),
        "pivot":      round(pivot, 2),
    }
 
 
def run_forecast(symbol):
    """
    Main forecast function.
    Returns dict with all forecast data for the dashboard.
    """
    df = get_historical_data(symbol, days=252)
 
    if df.empty or len(df) < 60:
        return None
 
    S0      = float(df["Close"].iloc[-1])
    returns = df["Daily_Return"].dropna()
    mu, sigma = estimate_params(returns)
 
    # Run simulations
    paths = run_gbm_simulations(S0, mu, sigma, T=HORIZON_DAYS, n_sims=N_SIMS)
 
    # Extract percentile bands
    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
 
    # Kalman smoothed historical
    smoothed_hist = kalman_smooth(df["Close"])
 
    # Support / resistance
    sr = support_resistance(df)
 
    # Scenarios
    final_prices  = paths[:, -1]
    bull_target   = float(np.percentile(final_prices, 75))
    base_target   = float(np.percentile(final_prices, 50))
    bear_target   = float(np.percentile(final_prices, 25))
 
    bull_ret = (bull_target - S0) / S0 * 100
    base_ret = (base_target - S0) / S0 * 100
    bear_ret = (bear_target - S0) / S0 * 100
 
    prob_up   = float((final_prices > S0).mean())
    prob_10up = float((final_prices > S0 * 1.10).mean())
    prob_10dn = float((final_prices < S0 * 0.90).mean())
 
    # Expected return and risk
    expected_ret = float(np.mean((final_prices - S0) / S0) * 100)
    cvar_5pct    = float(np.mean(
        (final_prices[final_prices < np.percentile(final_prices, 5)] - S0) / S0
    ) * 100)
 
    # Trend analysis from Kalman velocity
    velocities = []
    x = np.array([float(df["Close"].iloc[0]), 0.0])
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q, R = 1e-4, 1e-2
    for price in df["Close"].values:
        xp  = F @ x
        Pp  = F @ P @ F.T + Q * np.eye(2)
        S_k = float((H @ Pp @ H.T)[0, 0]) + R
        K   = (Pp @ H.T) / S_k
        Kf  = K.flatten()
        inn = float(price) - float((H @ xp)[0])
        x   = xp + Kf * inn
        P   = (np.eye(2) - Kf[:, None] * H) @ Pp
        velocities.append(float(x[1]))
 
    current_velocity = velocities[-1]
    vel_std = np.std(velocities[-60:])
    trend_strength = abs(current_velocity) / (vel_std + 1e-8)
 
    if current_velocity > 0 and trend_strength > 1.0:
        trend = "Strong Uptrend 📈"
        trend_color = "#00c853"
    elif current_velocity > 0:
        trend = "Mild Uptrend 📈"
        trend_color = "#69f0ae"
    elif current_velocity < 0 and trend_strength > 1.0:
        trend = "Strong Downtrend 📉"
        trend_color = "#ff1744"
    elif current_velocity < 0:
        trend = "Mild Downtrend 📉"
        trend_color = "#ff5252"
    else:
        trend = "Sideways ➡️"
        trend_color = "#ffa000"
 
    # Build date arrays
    hist_dates     = df["Date"].tolist()
    last_date      = pd.to_datetime(hist_dates[-1])
    future_dates   = pd.bdate_range(start=last_date, periods=HORIZON_DAYS + 1)[1:]
    future_dates_s = [str(d.date()) for d in future_dates]
 
    return {
        "symbol":          symbol,
        "current_price":   round(S0, 2),
        "mu_daily":        round(mu * 100, 4),
        "sigma_daily":     round(sigma * 100, 4),
        "hist_dates":      hist_dates,
        "hist_close":      df["Close"].tolist(),
        "smoothed_hist":   smoothed_hist,
        "future_dates":    future_dates_s,
        "p5":              p5.tolist(),
        "p25":             p25.tolist(),
        "p50":             p50.tolist(),
        "p75":             p75.tolist(),
        "p95":             p95.tolist(),
        "bull_target":     round(bull_target, 2),
        "base_target":     round(base_target, 2),
        "bear_target":     round(bear_target, 2),
        "bull_ret":        round(bull_ret, 2),
        "base_ret":        round(base_ret, 2),
        "bear_ret":        round(bear_ret, 2),
        "prob_up":         round(prob_up * 100, 1),
        "prob_10up":       round(prob_10up * 100, 1),
        "prob_10dn":       round(prob_10dn * 100, 1),
        "expected_ret":    round(expected_ret, 2),
        "cvar_5pct":       round(cvar_5pct, 2),
        "support":         sr["support"],
        "resistance":      sr["resistance"],
        "pivot":           sr["pivot"],
        "trend":           trend,
        "trend_color":     trend_color,
        "trend_strength":  round(trend_strength, 2),
        "n_sims":          N_SIMS,
        "horizon_days":    HORIZON_DAYS,
    }
 
 
if __name__ == "__main__":
    r = run_forecast("RELIANCE")
    if r:
        print(f"Symbol        : {r['symbol']}")
        print(f"Current Price : ₹{r['current_price']}")
        print(f"Trend         : {r['trend']}")
        print(f"Base Target   : ₹{r['base_target']} ({r['base_ret']:+.2f}%)")
        print(f"Bull Target   : ₹{r['bull_target']} ({r['bull_ret']:+.2f}%)")
        print(f"Bear Target   : ₹{r['bear_target']} ({r['bear_ret']:+.2f}%)")
        print(f"P(price up)   : {r['prob_up']}%")
        print(f"CVaR 5%       : {r['cvar_5pct']}%")
 