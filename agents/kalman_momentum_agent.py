"""
kalman_momentum_agent.py
------------------------
Advanced Momentum Agent using Kalman Filter + ADX.
 
Kalman Filter tracks price level and velocity simultaneously.
ADX measures trend strength — gates the signal when no trend exists.
 
Output: score between -1.0 and +1.0
"""
 
import numpy as np
import pandas as pd
import sqlite3
 
DB_PATH = "data/nifty50.db"
 
 
class KalmanFilter1D:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        self.Q = process_noise
        self.R = measurement_noise
        self.x = None
        self.P = np.eye(2) * 1.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
 
    def update(self, observation):
        if self.x is None:
            self.x = np.array([float(observation), 0.0])
 
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q * np.eye(2)
 
        S      = float((self.H @ P_pred @ self.H.T)[0, 0]) + self.R
        K      = (P_pred @ self.H.T) / S          # shape (2,1)
        K_flat = K.flatten()                       # shape (2,)
 
        innovation = float(observation) - float((self.H @ x_pred)[0])
        self.x = x_pred + K_flat * innovation
        self.P = (np.eye(2) - K_flat[:, None] * self.H) @ P_pred
 
        return float(self.x[0]), float(self.x[1]), float(K_flat[0])
 
    def filter_series(self, prices):
        filtered_prices, velocities, gains = [], [], []
        self.x = None
        self.P = np.eye(2) * 1.0
 
        for price in prices.values:
            fp, vel, gain = self.update(float(price))
            filtered_prices.append(fp)
            velocities.append(vel)
            gains.append(gain)
 
        return pd.DataFrame({
            "filtered_price": filtered_prices,
            "velocity":       velocities,
            "kalman_gain":    gains,
        }, index=prices.index)
 
 
def compute_adx(high, low, close, period=14):
    prev_close = close.shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
 
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
 
    plus_dm  = high - prev_high
    minus_dm = prev_low - low
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
 
    def wilder_smooth(series, n):
        result = series.copy() * 0.0
        if len(series) <= n:
            return result
        result.iloc[n] = series.iloc[1:n + 1].sum()
        for i in range(n + 1, len(series)):
            result.iloc[i] = result.iloc[i - 1] - result.iloc[i - 1] / n + series.iloc[i]
        return result
 
    atr      = wilder_smooth(tr, period)
    plus_di  = 100 * wilder_smooth(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = wilder_smooth(dx.fillna(0), period) / period
 
    return adx.fillna(0)
 
 
def compute_kalman_momentum_signal(df):
    close = df["Close"]
    high  = df["High"]  if "High" in df.columns else close
    low   = df["Low"]   if "Low"  in df.columns else close
 
    kf    = KalmanFilter1D(process_noise=1e-4, measurement_noise=1e-2)
    kf_df = kf.filter_series(close)
 
    velocity    = kf_df["velocity"]
    vel_std     = velocity.rolling(60).std().replace(0, np.nan)
    vel_norm    = (velocity / vel_std).clip(-2, 2) / 2
 
    adx         = compute_adx(high, low, close, period=14)
    adx_mult    = (adx / 25).clip(0.3, 1.0)
 
    signal = vel_norm * adx_mult
    return signal.clip(-1.0, 1.0)
 
 
def run_kalman_momentum_agent(symbol=None):
    conn  = sqlite3.connect(DB_PATH)
    query = (
        f"SELECT Date, Symbol, High, Low, Close FROM prices "
        f"WHERE Symbol='{symbol}' ORDER BY Date"
        if symbol else
        "SELECT Date, Symbol, High, Low, Close FROM prices ORDER BY Symbol, Date"
    )
    df = pd.read_sql(query, conn)
    conn.close()
 
    results = []
    for sym, group in df.groupby("Symbol"):
        group = group.copy().set_index("Date")
        group["Kalman_Momentum_Score"] = compute_kalman_momentum_signal(group)
        results.append(group.reset_index()[["Date", "Symbol", "Kalman_Momentum_Score"]])
 
    return pd.concat(results, ignore_index=True)
 
 
def get_latest_signal(symbol):
    df    = run_kalman_momentum_agent(symbol=symbol).dropna()
    if df.empty:
        return {"symbol": symbol, "score": 0.0, "interpretation": "no data",
                "agent": "kalman_momentum"}
 
    latest = df.iloc[-1]
    score  = round(float(latest["Kalman_Momentum_Score"]), 4)
 
    if score > 0.4:    interpretation = "strong uptrend (Kalman confirmed)"
    elif score > 0.1:  interpretation = "mild uptrend"
    elif score < -0.4: interpretation = "strong downtrend (Kalman confirmed)"
    elif score < -0.1: interpretation = "mild downtrend"
    else:              interpretation = "no clear trend — ADX low"
 
    return {
        "symbol": symbol, "agent": "kalman_momentum",
        "score": score, "interpretation": interpretation,
        "date": latest["Date"],
    }
 
 
if __name__ == "__main__":
    print("Kalman Momentum Agent test\n")
    for sym in ["RELIANCE", "TCS", "HDFCBANK"]:
        r = get_latest_signal(sym)
        print(f"{sym:15} score={r['score']:+.4f}  {r['interpretation']}")