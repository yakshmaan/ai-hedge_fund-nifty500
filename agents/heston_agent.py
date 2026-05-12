"""
heston_agent.py
---------------
Heston Stochastic Volatility Agent.

GBM assumes constant volatility. Heston fixes this:
  dS = μS dt + √v S dW₁
  dv = κ(θ-v)dt + ξ√v dW₂

Where:
  v  = stochastic variance (changes over time)
  κ  = mean reversion speed of variance
  θ  = long-run average variance
  ξ  = volatility of volatility (vol-of-vol)
  ρ  = correlation between price and variance shocks

Key insight: when volatility is ABOVE its long-run average,
it tends to fall → price distribution has fatter tails.
When volatility is BELOW average, it tends to rise.
This gives better tail risk estimates than GBM.

Signal logic:
  - If current vol >> long-run vol → high risk → bearish adjustment
  - If current vol << long-run vol → calm market → bullish boost
  - Vol-of-vol (ξ) measures uncertainty about uncertainty

Output: score between -1.0 and +1.0
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings("ignore")

DB_PATH      = "data/nifty50.db"
N_SIMS       = 5000
HORIZON      = 5


def estimate_heston_params(returns, window=120):
    """
    Estimate Heston model parameters from historical returns.

    κ  = speed of variance mean reversion (estimated from vol autocorrelation)
    θ  = long-run variance (historical variance)
    ξ  = vol of vol (std of rolling variance)
    ρ  = price-vol correlation
    v0 = current variance
    """
    ret = returns.dropna().iloc[-window:]
    if len(ret) < 60:
        return None

    # Rolling variance (21-day)
    rolling_var = ret.rolling(21).var().dropna()

    if len(rolling_var) < 30:
        return None

    # Long-run variance θ
    theta = float(rolling_var.mean())

    # Current variance v0
    v0 = float(rolling_var.iloc[-1])

    # Vol of vol ξ — std of rolling variance
    xi = float(rolling_var.std())

    # Mean reversion speed κ — from AR(1) of variance
    var_lag  = rolling_var.iloc[:-1].values
    var_curr = rolling_var.iloc[1:].values
    if len(var_lag) > 10:
        try:
            X    = np.column_stack([var_lag, np.ones(len(var_lag))])
            beta = np.linalg.lstsq(X, var_curr, rcond=None)[0]
            kappa = float(max(1 - beta[0], 0.1))  # AR coefficient → mean reversion speed
        except Exception:
            kappa = 2.0
    else:
        kappa = 2.0

    # Price-vol correlation ρ
    price_ret  = ret.values
    vol_change = np.diff(rolling_var.values)
    min_len    = min(len(price_ret) - 1, len(vol_change))
    if min_len > 10:
        rho = float(np.corrcoef(price_ret[-min_len:], vol_change[-min_len:])[0, 1])
        rho = float(np.clip(rho, -0.99, 0.99))
    else:
        rho = -0.7  # typical negative correlation (leverage effect)

    # Drift μ
    mu = float(ret.mean())

    return {
        "mu":    mu,
        "kappa": kappa,
        "theta": theta,
        "xi":    xi,
        "rho":   rho,
        "v0":    v0,
    }


def simulate_heston(S0, params, T=HORIZON, n_sims=N_SIMS):
    """
    Simulate Heston model price paths.

    Uses Euler-Maruyama discretization:
      S(t+dt) = S(t) * exp((μ - v/2)*dt + √v * √dt * Z₁)
      v(t+dt) = v(t) + κ(θ-v)*dt + ξ*√v*√dt * Z₂
      Z₁, Z₂ are correlated: corr(Z₁,Z₂) = ρ
    """
    mu    = params["mu"]
    kappa = params["kappa"]
    theta = params["theta"]
    xi    = params["xi"]
    rho   = params["rho"]
    v0    = params["v0"]

    dt    = 1.0
    rng   = np.random.default_rng(42)

    S = np.full(n_sims, S0)
    v = np.full(n_sims, max(v0, 1e-6))

    for t in range(T):
        Z1 = rng.standard_normal(n_sims)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_sims)

        # Variance update (full truncation scheme — prevents negative variance)
        v_pos  = np.maximum(v, 0)
        v_new  = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * Z2
        v      = np.maximum(v_new, 1e-8)

        # Price update
        S = S * np.exp((mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * Z1)

    return S


def compute_heston_signal(S0, final_prices, current_v, theta):
    """
    Convert Heston simulation results into a signal score.

    Score components:
      1. Probability of gain (like GBM agent)
      2. Volatility regime: is current vol above or below long-run?
         High vol regime → bearish adjustment (risk-off)
         Low vol regime  → bullish boost (risk-on)
      3. Tail risk: CVaR vs expected return ratio
    """
    returns = (final_prices - S0) / S0

    prob_gain    = float((final_prices > S0).mean())
    expected_ret = float(returns.mean())
    cvar_5       = float(returns[returns <= np.percentile(returns, 5)].mean())

    # Volatility regime signal
    vol_ratio = current_v / (theta + 1e-8)
    if vol_ratio > 1.5:
        vol_signal = -0.3   # high vol → bearish
    elif vol_ratio > 1.2:
        vol_signal = -0.15
    elif vol_ratio < 0.7:
        vol_signal = +0.2   # low vol → bullish
    elif vol_ratio < 0.85:
        vol_signal = +0.1
    else:
        vol_signal = 0.0

    # Probability component
    prob_component = (prob_gain - 0.5) * 2

    # Risk-adjusted component
    if abs(cvar_5) > 1e-8:
        risk_adj = expected_ret / abs(cvar_5)
        risk_component = float(np.clip(risk_adj * 0.5, -0.5, 0.5))
    else:
        risk_component = 0.0

    score = 0.4 * prob_component + 0.3 * vol_signal + 0.3 * risk_component
    return float(np.clip(score, -1, 1))


def get_latest_signal(symbol):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT Date, Close, Daily_Return FROM prices WHERE Symbol='{symbol}' ORDER BY Date",
        conn, index_col="Date"
    )
    conn.close()

    if len(df) < 150:
        return {"symbol": symbol, "agent": "heston", "score": 0.0,
                "interpretation": "insufficient data"}

    S0      = float(df["Close"].iloc[-1])
    returns = df["Daily_Return"].dropna()
    params  = estimate_heston_params(returns)

    if params is None:
        return {"symbol": symbol, "agent": "heston", "score": 0.0,
                "interpretation": "parameter estimation failed"}

    final_prices = simulate_heston(S0, params, T=HORIZON, n_sims=N_SIMS)
    score        = compute_heston_signal(S0, final_prices, params["v0"], params["theta"])
    score        = round(score, 4)

    vol_ratio = params["v0"] / (params["theta"] + 1e-8)
    if score > 0.3:    interp = f"Heston: bullish | vol_ratio={vol_ratio:.2f} (ρ={params['rho']:.2f})"
    elif score < -0.3: interp = f"Heston: bearish | vol_ratio={vol_ratio:.2f} (ρ={params['rho']:.2f})"
    else:              interp = f"Heston: neutral | vol_ratio={vol_ratio:.2f} (ρ={params['rho']:.2f})"

    return {
        "symbol":         symbol,
        "agent":          "heston",
        "score":          score,
        "interpretation": interp,
        "vol_ratio":      round(vol_ratio, 3),
        "rho":            round(params["rho"], 3),
        "kappa":          round(params["kappa"], 3),
        "theta":          round(params["theta"], 6),
        "v0":             round(params["v0"], 6),
        "date":           str(df.index[-1]),
    }


if __name__ == "__main__":
    print(f"Heston Stochastic Volatility Agent ({N_SIMS:,} sims)\n")
    for sym in ["RELIANCE", "TCS", "HDFCBANK"]:
        r = get_latest_signal(sym)
        print(f"{sym:15} score={r['score']:+.4f}  {r['interpretation']}")
