"""
gbm_monte_carlo_agent.py
------------------------
Monte Carlo agent using Geometric Brownian Motion (GBM).
 
Instead of asking "is this stock going up or down?" (binary),
this agent asks "what is the PROBABILITY DISTRIBUTION of prices
over the next N days?" — a much richer question.
 
Geometric Brownian Motion models stock price as:
    dS = μ·S·dt + σ·S·dW
 
Where:
    S  = stock price
    μ  = drift (expected return per day)
    σ  = volatility (standard deviation of returns)
    dW = Wiener process (random shock ~ N(0, dt))
 
Discrete form for simulation:
    S(t+1) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z)
    where Z ~ N(0,1)
 
We run 10,000 simulations of the next 5 trading days.
This gives us a full probability distribution of future prices.
 
From that distribution we extract:
    - P(price > current)  → probability of gain
    - Expected return     → mean of simulated returns
    - CVaR (5%)          → expected loss in worst 5% scenarios
    - Score               → risk-adjusted probability signal
 
Usage:
    python agents/gbm_monte_carlo_agent.py
"""
 
import numpy as np
import pandas as pd
import sqlite3
 
DB_PATH      = "data/nifty50.db"
N_SIMULATIONS = 10_000
HORIZON_DAYS  = 5       # predict 5 trading days ahead
RISK_FREE     = 0.06 / 252  # daily risk-free rate (6% annual, India)
 
 
# ── GBM SIMULATION ────────────────────────────────────────────────────────────
 
def estimate_gbm_params(returns: pd.Series, window: int = 60) -> tuple:
    """
    Estimate GBM parameters from recent returns.
 
    Uses last `window` days to estimate:
        mu    = mean daily log return (drift)
        sigma = std of daily log returns (volatility)
 
    Returns (mu, sigma) as daily values.
    """
    recent = returns.iloc[-window:].dropna()
 
    if len(recent) < 20:
        return 0.0, 0.02  # fallback: zero drift, 2% daily vol
 
    log_returns = np.log(1 + recent.replace(-1, np.nan).dropna())
    mu    = float(log_returns.mean())
    sigma = float(log_returns.std())
 
    return mu, max(sigma, 1e-6)
 
 
def simulate_gbm(S0: float, mu: float, sigma: float,
                 T: int = HORIZON_DAYS,
                 n_sims: int = N_SIMULATIONS,
                 seed: int = 42) -> np.ndarray:
    """
    Simulate N_SIMULATIONS price paths over T days using GBM.
 
    Returns: array of shape (n_sims,) with final prices after T days.
 
    The GBM formula in discrete time:
        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0  # one day at a time
 
    # Shape: (n_sims, T) — each row is one price path
    Z = rng.standard_normal((n_sims, T))
 
    # Log return for each step
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
 
    # Cumulative product gives price path
    # Price at time T = S0 * exp(sum of log returns)
    final_prices = S0 * np.exp(log_returns.sum(axis=1))
 
    return final_prices
 
 
# ── RISK METRICS FROM SIMULATION ──────────────────────────────────────────────
 
def compute_simulation_metrics(S0: float, final_prices: np.ndarray) -> dict:
    """
    Extract useful statistics from the Monte Carlo simulation results.
 
    Returns:
        prob_gain:     P(final price > current price)
        expected_ret:  mean expected return over horizon
        cvar_5pct:     Conditional VaR — average loss in worst 5% scenarios
        sharpe_mc:     Simplified Sharpe from simulation
        score:         Final signal score (-1 to +1)
    """
    returns = (final_prices - S0) / S0
 
    prob_gain    = float((final_prices > S0).mean())
    expected_ret = float(returns.mean())
    vol          = float(returns.std())
 
    # CVaR (Expected Shortfall) — average loss in worst 5%
    worst_5pct = np.percentile(returns, 5)
    cvar       = float(returns[returns <= worst_5pct].mean())
 
    # Sharpe (simplified, over horizon)
    rf_horizon = RISK_FREE * HORIZON_DAYS
    sharpe_mc  = (expected_ret - rf_horizon) / (vol + 1e-8)
 
    # Score: combines probability of gain and Sharpe
    # Center prob_gain at 0.5 → range -0.5 to +0.5
    # Add Sharpe contribution (clipped)
    prob_component   = (prob_gain - 0.5) * 2           # -1 to +1
    sharpe_component = np.clip(sharpe_mc / 2, -0.5, 0.5)
 
    score = 0.6 * prob_component + 0.4 * sharpe_component
    score = float(np.clip(score, -1, 1))
 
    return {
        "prob_gain":    round(prob_gain, 4),
        "expected_ret": round(expected_ret * 100, 4),  # as %
        "cvar_5pct":    round(cvar * 100, 4),          # as %
        "volatility":   round(vol * 100, 4),           # as %
        "sharpe_mc":    round(sharpe_mc, 4),
        "score":        round(score, 4),
    }
 
 
# ── AGENT INTERFACE ───────────────────────────────────────────────────────────
 
def get_latest_signal(symbol: str) -> dict:
    """
    Run GBM Monte Carlo for a stock and return signal.
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT Date, Close, Daily_Return FROM prices "
        f"WHERE Symbol='{symbol}' ORDER BY Date",
        conn, index_col="Date"
    )
    conn.close()
 
    if len(df) < 60:
        return {"symbol": symbol, "score": 0.0, "interpretation": "insufficient data"}
 
    S0      = float(df["Close"].iloc[-1])
    returns = df["Daily_Return"].dropna()
 
    # Estimate GBM parameters from recent data
    mu, sigma = estimate_gbm_params(returns, window=60)
 
    # Run Monte Carlo simulation
    final_prices = simulate_gbm(S0, mu, sigma, T=HORIZON_DAYS, n_sims=N_SIMULATIONS)
 
    # Compute metrics
    metrics = compute_simulation_metrics(S0, final_prices)
    score   = metrics["score"]
 
    # Build interpretation
    if metrics["prob_gain"] > 0.60:
        interp = f"MC: {metrics['prob_gain']:.0%} prob gain | E[ret]={metrics['expected_ret']:+.2f}%"
    elif metrics["prob_gain"] < 0.40:
        interp = f"MC: {metrics['prob_gain']:.0%} prob gain | CVaR={metrics['cvar_5pct']:.2f}%"
    else:
        interp = f"MC: near-even odds | σ={metrics['volatility']:.2f}%/5d"
 
    return {
        "symbol":       symbol,
        "agent":        "gbm_monte_carlo",
        "score":        score,
        "interpretation": interp,
        "prob_gain":    metrics["prob_gain"],
        "expected_ret": metrics["expected_ret"],
        "cvar_5pct":    metrics["cvar_5pct"],
        "sharpe_mc":    metrics["sharpe_mc"],
        "current_price": S0,
        "date":         str(df.index[-1]),
    }
 
 
def run_portfolio_monte_carlo(symbols: list, weights: dict) -> dict:
    """
    Run portfolio-level Monte Carlo simulation.
    Simulates correlated asset paths and returns portfolio distribution.
 
    Used by risk engine to get portfolio CVaR.
    """
    conn  = sqlite3.connect(DB_PATH)
    sims  = {}
 
    for sym in symbols:
        df = pd.read_sql(
            f"SELECT Date, Close, Daily_Return FROM prices "
            f"WHERE Symbol='{sym}' ORDER BY Date",
            conn, index_col="Date"
        )
        if len(df) < 60:
            continue
 
        S0        = float(df["Close"].iloc[-1])
        returns   = df["Daily_Return"].dropna()
        mu, sigma = estimate_gbm_params(returns, window=60)
        final     = simulate_gbm(S0, mu, sigma, T=HORIZON_DAYS, n_sims=N_SIMULATIONS)
        sims[sym] = (final - S0) / S0  # store as returns
 
    conn.close()
 
    if not sims:
        return {"portfolio_cvar": 0.0, "portfolio_expected_ret": 0.0}
 
    # Weighted portfolio return across all simulations
    portfolio_returns = np.zeros(N_SIMULATIONS)
    total_weight      = 0.0
 
    for sym, ret_sim in sims.items():
        w = weights.get(sym, 1.0 / len(sims))
        portfolio_returns += w * ret_sim
        total_weight      += w
 
    if total_weight > 0:
        portfolio_returns /= total_weight
 
    cvar_threshold   = np.percentile(portfolio_returns, 5)
    portfolio_cvar   = float(portfolio_returns[portfolio_returns <= cvar_threshold].mean())
    expected_ret     = float(portfolio_returns.mean())
 
    return {
        "portfolio_cvar":         round(portfolio_cvar * 100, 4),
        "portfolio_expected_ret": round(expected_ret * 100, 4),
        "prob_portfolio_gain":    round(float((portfolio_returns > 0).mean()), 4),
    }
 
 
if __name__ == "__main__":
    print(f"GBM Monte Carlo Agent ({N_SIMULATIONS:,} simulations, {HORIZON_DAYS}-day horizon)\n")
 
    for sym in ["RELIANCE", "TCS", "HDFCBANK"]:
        r = get_latest_signal(sym)
        print(f"{sym:15} score={r['score']:+.4f} | "
              f"P(gain)={r['prob_gain']:.1%} | "
              f"E[ret]={r['expected_ret']:+.2f}% | "
              f"CVaR={r['cvar_5pct']:.2f}%")
 
    print("\nPortfolio Monte Carlo test:")
    port = run_portfolio_monte_carlo(
        ["RELIANCE", "TCS", "HDFCBANK"],
        {"RELIANCE": 0.4, "TCS": 0.3, "HDFCBANK": 0.3}
    )
    print(f"  Portfolio E[ret]: {port['portfolio_expected_ret']:+.2f}%")
    print(f"  Portfolio CVaR:   {port['portfolio_cvar']:.2f}%")
    print(f"  P(portfolio gain): {port['prob_portfolio_gain']:.1%}")
 