"""
momentum_agent.py
-----------------
Momentum Agent: looks at price trends and outputs a signal score.

Logic:
  - Uses two moving averages: fast (20-day) and slow (50-day)
  - If fast MA is above slow MA → uptrend → positive score
  - If fast MA is below slow MA → downtrend → negative score
  - Score is scaled by how far apart the MAs are (stronger trend = stronger signal)

Output: score between -1.0 and +1.0 for each stock on each date

This is the Moving Average Crossover strategy from your research paper,
but now packaged as a reusable agent that plugs into the orchestrator.

Usage:
    python agents/momentum_agent.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os

DB_PATH = "data/nifty50.db"

# ── CORE LOGIC ────────────────────────────────────────────────────────────────

def compute_momentum_signal(df: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame with a 'Close' column,
    compute a momentum signal score for each row.

    Returns a Series of scores between -1.0 and +1.0.
    """
    close = df["Close"]

    # Fast and slow moving averages
    ma_fast = close.rolling(window=20).mean()   # 20-day MA
    ma_slow = close.rolling(window=50).mean()   # 50-day MA

    # Raw difference as % of slow MA
    # Positive = fast above slow (uptrend), Negative = fast below slow (downtrend)
    raw_signal = (ma_fast - ma_slow) / ma_slow

    # Also compute RSI (Relative Strength Index) as a second momentum indicator
    # RSI > 70 = overbought, RSI < 30 = oversold
    rsi = compute_rsi(close, period=14)

    # Normalize RSI to -1 to +1 scale
    # RSI of 50 = neutral (0), RSI of 100 = max bullish (+1), RSI of 0 = max bearish (-1)
    rsi_signal = (rsi - 50) / 50

    # Combine: 60% weight on MA crossover, 40% on RSI
    combined = (0.6 * raw_signal) + (0.4 * rsi_signal)

    # Clip to -1 to +1 range
    score = combined.clip(-1.0, 1.0)

    return score


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index).
    RSI measures speed and magnitude of recent price changes.
    Returns values between 0 and 100.
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)  # fill NaN with neutral 50


# ── RUN ON ALL STOCKS ─────────────────────────────────────────────────────────

def run_momentum_agent(symbol: str = None) -> pd.DataFrame:
    """
    Run momentum agent on one stock (or all stocks if symbol=None).
    Returns DataFrame with columns: Date, Symbol, Momentum_Score
    """
    conn = sqlite3.connect(DB_PATH)

    if symbol:
        query = f"SELECT Date, Symbol, Close FROM prices WHERE Symbol = '{symbol}' ORDER BY Date"
    else:
        query = "SELECT Date, Symbol, Close FROM prices ORDER BY Symbol, Date"

    df = pd.read_sql(query, conn)
    conn.close()

    results = []

    for sym, group in df.groupby("Symbol"):
        group = group.copy().set_index("Date")
        group["Momentum_Score"] = compute_momentum_signal(group)
        group = group.reset_index()[["Date", "Symbol", "Momentum_Score"]]
        results.append(group)

    result_df = pd.concat(results, ignore_index=True)
    return result_df


def get_latest_signal(symbol: str) -> dict:
    """
    Get the most recent momentum signal for a single stock.
    This is what the orchestrator will call in Phase 4.

    Returns: {"symbol": ..., "score": ..., "interpretation": ...}
    """
    df = run_momentum_agent(symbol=symbol)
    df = df.dropna()

    if df.empty:
        return {"symbol": symbol, "score": 0.0, "interpretation": "no data"}

    latest = df.iloc[-1]
    score = round(latest["Momentum_Score"], 4)

    if score > 0.3:
        interpretation = "strong uptrend"
    elif score > 0.05:
        interpretation = "mild uptrend"
    elif score < -0.3:
        interpretation = "strong downtrend"
    elif score < -0.05:
        interpretation = "mild downtrend"
    else:
        interpretation = "neutral / sideways"

    return {
        "symbol": symbol,
        "agent": "momentum",
        "score": score,
        "interpretation": interpretation,
        "date": latest["Date"],
    }


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Momentum Agent on all Nifty 50 stocks...\n")

    signals = run_momentum_agent()
    signals = signals.dropna()

    # Show latest signal per stock
    latest = signals.groupby("Symbol").last().reset_index()
    latest = latest.sort_values("Momentum_Score", ascending=False)

    print("Top 5 bullish stocks (momentum):")
    print(latest.head(5).to_string(index=False))

    print("\nTop 5 bearish stocks (momentum):")
    print(latest.tail(5).to_string(index=False))

    # Test get_latest_signal
    print("\nSingle stock test:")
    print(get_latest_signal("RELIANCE"))