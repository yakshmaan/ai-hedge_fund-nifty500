"""
mean_reversion_agent.py
-----------------------
Mean Reversion Agent: identifies overbought and oversold conditions.

Core idea:
  Prices tend to revert to their average over time.
  If a stock has gone up too far too fast → likely to come back down → sell signal
  If a stock has fallen too far too fast → likely to bounce up → buy signal

This is the OPPOSITE logic to momentum. That's intentional.
When combined in Phase 4, momentum + mean reversion together
create a more balanced view than either alone.

Indicators used:
  - Bollinger Bands: price vs its own rolling mean ± 2 standard deviations
  - Z-Score: how many standard deviations from the rolling mean is the current price

Output: score between -1.0 and +1.0 for each stock on each date

Usage:
    python agents/mean_reversion_agent.py
"""

import pandas as pd
import numpy as np
import sqlite3

DB_PATH = "data/nifty50.db"

# ── CORE LOGIC ────────────────────────────────────────────────────────────────

def compute_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-Score: how far is the current price from its rolling average,
    measured in standard deviations.

    Z = (price - rolling_mean) / rolling_std

    Z > +2  → price is very high vs recent history → likely to fall → negative signal
    Z < -2  → price is very low vs recent history  → likely to rise → positive signal
    """
    rolling_mean = close.rolling(window=window).mean()
    rolling_std  = close.rolling(window=window).std()

    zscore = (close - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore


def compute_bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Where is the price within its Bollinger Bands?

    Bollinger Bands = rolling mean ± 2 standard deviations

    Returns a value:
      +1.0 = price is AT the upper band (overbought)
       0.0 = price is at the middle (neutral)
      -1.0 = price is AT the lower band (oversold)
    """
    rolling_mean = close.rolling(window=window).mean()
    rolling_std  = close.rolling(window=window).std()

    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    band_width = upper_band - lower_band

    # Position within band: 0 = at lower band, 1 = at upper band
    position = (close - lower_band) / band_width.replace(0, np.nan)

    # Rescale to -1 to +1 (0.5 = middle = neutral)
    position_scaled = (position - 0.5) * 2

    return position_scaled.clip(-1.0, 1.0)


def compute_mean_reversion_signal(df: pd.DataFrame) -> pd.Series:
    """
    Combine Z-Score and Bollinger Band position into one mean reversion signal.

    NOTE: The signal is INVERTED from the raw indicators.
    High Z-Score → overbought → we SELL → negative score
    Low Z-Score  → oversold  → we BUY  → positive score
    """
    close = df["Close"]

    zscore = compute_zscore(close, window=20)
    bb_pos = compute_bollinger_position(close, window=20)

    # Normalize z-score to -1 to +1 range (clip beyond 3 std)
    zscore_normalized = (zscore / 3).clip(-1.0, 1.0)

    # Combine: 50% z-score, 50% bollinger position
    raw_signal = (0.5 * zscore_normalized) + (0.5 * bb_pos)

    # INVERT: high = overbought = sell signal (negative)
    score = -raw_signal

    return score.clip(-1.0, 1.0)


# ── RUN ON ALL STOCKS ─────────────────────────────────────────────────────────

def run_mean_reversion_agent(symbol: str = None) -> pd.DataFrame:
    """
    Run mean reversion agent on one or all stocks.
    Returns DataFrame with: Date, Symbol, MeanReversion_Score
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
        group["MeanReversion_Score"] = compute_mean_reversion_signal(group)
        group = group.reset_index()[["Date", "Symbol", "MeanReversion_Score"]]
        results.append(group)

    return pd.concat(results, ignore_index=True)


def get_latest_signal(symbol: str) -> dict:
    """
    Get the most recent mean reversion signal for a single stock.
    Called by the orchestrator in Phase 4.
    """
    df = run_mean_reversion_agent(symbol=symbol)
    df = df.dropna()

    if df.empty:
        return {"symbol": symbol, "score": 0.0, "interpretation": "no data"}

    latest = df.iloc[-1]
    score  = round(latest["MeanReversion_Score"], 4)

    if score > 0.4:
        interpretation = "strongly oversold — likely to bounce"
    elif score > 0.1:
        interpretation = "mildly oversold"
    elif score < -0.4:
        interpretation = "strongly overbought — likely to fall"
    elif score < -0.1:
        interpretation = "mildly overbought"
    else:
        interpretation = "price near its mean — neutral"

    return {
        "symbol": symbol,
        "agent": "mean_reversion",
        "score": score,
        "interpretation": interpretation,
        "date": latest["Date"],
    }


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Mean Reversion Agent on all Nifty 50 stocks...\n")

    signals = run_mean_reversion_agent()
    signals = signals.dropna()

    latest = signals.groupby("Symbol").last().reset_index()
    latest = latest.sort_values("MeanReversion_Score", ascending=False)

    print("Top 5 oversold stocks (mean reversion buy signals):")
    print(latest.head(5).to_string(index=False))

    print("\nTop 5 overbought stocks (mean reversion sell signals):")
    print(latest.tail(5).to_string(index=False))

    print("\nSingle stock test:")
    print(get_latest_signal("TCS"))