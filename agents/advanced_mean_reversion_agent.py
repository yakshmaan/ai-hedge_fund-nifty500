"""
advanced_mean_reversion_agent.py
---------------------------------
Advanced Mean Reversion Agent using:
  1. ADF Test     — only trade mean reversion on statistically stationary series
  2. Hurst Exponent — measures how strongly a series mean-reverts
  3. Pairs Cointegration — find stock pairs that move together, trade the spread
"""
 
import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings("ignore")
 
DB_PATH = "data/nifty50.db"
 
COINTEGRATED_PAIRS = [
    ("HDFCBANK", "ICICIBANK"),
    ("INFY",     "TCS"),
    ("HINDUNILVR", "NESTLEIND"),
    ("AXISBANK", "KOTAKBANK"),
    ("WIPRO",    "HCLTECH"),
]
 
 
def adf_test(series, max_lags=5):
    y = series.dropna().values.astype(float)
    n = len(y)
    if n < 20:
        return {"statistic": 0, "p_value": 1.0, "is_stationary": False}
    dy    = np.diff(y)
    y_lag = y[:-1]
    lags  = min(max_lags, int(np.floor(12 * (n / 100) ** 0.25)))
    X_cols = [y_lag[lags:]]
    for lag in range(1, lags + 1):
        X_cols.append(dy[lags - lag: len(dy) - lag + (0 if lag < lags else 0)])
    min_len = min(len(c) for c in X_cols)
    X_cols  = [c[:min_len] for c in X_cols]
    X_cols.append(np.ones(min_len))
    X = np.column_stack(X_cols)
    Y = dy[lags:lags + min_len]
    if len(Y) < 10:
        return {"statistic": 0, "p_value": 1.0, "is_stationary": False}
    try:
        beta     = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid    = Y - X @ beta
        sigma2   = resid.var()
        XtX_inv  = np.linalg.inv(X.T @ X)
        se       = np.sqrt(sigma2 * XtX_inv[0, 0])
        adf_stat = beta[0] / (se + 1e-10)
    except Exception:
        return {"statistic": 0, "p_value": 1.0, "is_stationary": False}
    if adf_stat < -3.43:   p_value = 0.01
    elif adf_stat < -2.86: p_value = 0.05
    elif adf_stat < -2.57: p_value = 0.10
    else:                  p_value = 0.50
    return {"statistic": round(adf_stat, 4), "p_value": p_value, "is_stationary": p_value <= 0.05}
 
 
def hurst_exponent(series, max_lag=100):
    ts = series.dropna().values
    n  = len(ts)
    if n < 50:
        return 0.5
    lags      = range(2, min(max_lag, n // 4))
    rs_values = []
    for lag in lags:
        rs_lag = []
        for start in range(0, n - lag, lag):
            chunk = ts[start: start + lag]
            mean  = chunk.mean()
            dev   = np.cumsum(chunk - mean)
            r     = dev.max() - dev.min()
            s     = chunk.std()
            if s > 0:
                rs_lag.append(r / s)
        if rs_lag:
            rs_values.append(np.mean(rs_lag))
    if len(rs_values) < 5:
        return 0.5
    try:
        H = np.polyfit(np.log(list(lags)[:len(rs_values)]), np.log(rs_values), 1)[0]
        return float(np.clip(H, 0.0, 1.0))
    except Exception:
        return 0.5
 
 
def test_cointegration(y1, y2):
    aligned = pd.concat([y1, y2], axis=1).dropna()
    if len(aligned) < 60:
        return {"cointegrated": False}
    a    = aligned.iloc[:, 0].values
    b    = aligned.iloc[:, 1].values
    X    = np.column_stack([np.ones(len(b)), b])
    beta = np.linalg.lstsq(X, a, rcond=None)[0]
    spread = a - (beta[0] + beta[1] * b)
    adf  = adf_test(pd.Series(spread))
    return {
        "cointegrated": adf["is_stationary"],
        "hedge_ratio":  round(beta[1], 4),
        "spread_mean":  round(spread.mean(), 4),
        "spread_std":   round(spread.std(), 4),
        "spread":       spread,
    }
 
 
def compute_advanced_mr_signal(df):
    close     = df["Close"]
    log_close = np.log(close.replace(0, np.nan).ffill())
 
    def rolling_adf_stat(series, window=120):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            result.iloc[i] = adf_test(series.iloc[i - window:i])["statistic"]
        return result
 
    adf_stats = rolling_adf_stat(log_close)
    adf_gate  = (adf_stats < -2.0).astype(float).fillna(0)
 
    hurst = close.rolling(200).apply(lambda x: hurst_exponent(pd.Series(x)), raw=False).fillna(0.5)
    hurst_mult = ((0.5 - hurst) * 4).clip(0, 1)
 
    rolling_mean  = close.rolling(20).mean()
    rolling_std   = close.rolling(20).std()
    zscore        = (close - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore_signal = -(zscore / 3).clip(-1, 1)
 
    signal = zscore_signal * adf_gate * (0.5 + 0.5 * hurst_mult)
    return signal.clip(-1, 1).fillna(0)
 
 
def get_pairs_signal(symbol, prices_df):
    for s1, s2 in COINTEGRATED_PAIRS:
        partner = None
        if symbol == s1: partner = s2
        if symbol == s2: partner = s1
        if partner is None or partner not in prices_df.columns:
            continue
        coint = test_cointegration(prices_df[symbol].dropna(), prices_df[partner].dropna())
        if not coint["cointegrated"] or coint["spread_std"] == 0:
            continue
        z = (coint["spread"][-1] - coint["spread_mean"]) / coint["spread_std"]
        if z > 1.5:   return -1.0 * min(abs(z) / 3, 1.0)
        elif z < -1.5: return  1.0 * min(abs(z) / 3, 1.0)
    return 0.0
 
 
def get_latest_signal(symbol):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT Date, Close FROM prices WHERE Symbol='{symbol}' ORDER BY Date",
        conn, index_col="Date"
    )
    all_prices = pd.read_sql(
        "SELECT Date, Symbol, Close FROM prices ORDER BY Date", conn
    ).pivot(index="Date", columns="Symbol", values="Close")
    conn.close()
 
    if len(df) < 150:
        return {"symbol": symbol, "score": 0.0, "interpretation": "insufficient data",
                "hurst": 0.5, "pairs_score": 0.0, "agent": "advanced_mean_reversion"}
 
    single_score = float(compute_advanced_mr_signal(df).iloc[-1])
    if np.isnan(single_score):
        single_score = 0.0
 
    pairs_score = get_pairs_signal(symbol, all_prices)
    H           = hurst_exponent(df["Close"])
    combined    = float(np.clip(0.7 * single_score + 0.3 * pairs_score, -1, 1))
    score       = round(combined, 4)
 
    if H < 0.4:   hurst_desc = f"strongly mean-reverting (H={H:.2f})"
    elif H < 0.5: hurst_desc = f"mildly mean-reverting (H={H:.2f})"
    elif H < 0.6: hurst_desc = f"near random walk (H={H:.2f})"
    else:         hurst_desc = f"trending series (H={H:.2f}) — MR unreliable"
 
    if score > 0.3:    interpretation = f"oversold signal | {hurst_desc}"
    elif score < -0.3: interpretation = f"overbought signal | {hurst_desc}"
    else:              interpretation = f"neutral | {hurst_desc}"
 
    return {
        "symbol": symbol, "agent": "advanced_mean_reversion",
        "score": score, "interpretation": interpretation,
        "hurst": round(H, 3), "pairs_score": round(pairs_score, 4),
        "date": str(df.index[-1]),
    }
 
 
if __name__ == "__main__":
    print("Advanced Mean Reversion Agent test\n")
    for sym in ["HDFCBANK", "TCS", "RELIANCE"]:
        print(f"Testing {sym}...")
        r = get_latest_signal(sym)
        print(f"{sym:15} score={r['score']:+.4f}  H={r['hurst']:.3f}  {r['interpretation']}\n")