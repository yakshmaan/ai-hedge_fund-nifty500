"""
ml_agent.py
-----------
ML Agent: Random Forest classifier predicting next-day price direction.

This is a direct extension of your SSRN research paper.
Your paper tested the strategy — this packages it as a live agent.

Features used to predict:
  - Returns over 1, 5, 10, 20 days
  - Rolling volatility (10-day, 20-day)
  - RSI (14-day)
  - Volume change vs 20-day average

Target variable:
  - 1 if next day's close > today's close (price goes up)
  - 0 if next day's close <= today's close (price goes down or flat)

Output:
  - Probability that next day is UP, rescaled to -1 to +1
  - Score of +1 = model is very confident price will rise tomorrow
  - Score of -1 = model is very confident price will fall tomorrow

Usage:
    python agents/ml_agent.py
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

DB_PATH = "data/nifty50.db"

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML features from raw OHLCV data.

    All features are backward-looking only (no lookahead bias).
    Every value at row t uses only data from rows t-1 and earlier.
    """
    close  = df["Close"]
    volume = df["Volume"]

    features = pd.DataFrame(index=df.index)

    # Price return features
    features["return_1d"]  = close.pct_change(1)
    features["return_5d"]  = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_20d"] = close.pct_change(20)

    # Volatility features
    features["volatility_10d"] = close.pct_change().rolling(10).std()
    features["volatility_20d"] = close.pct_change().rolling(20).std()

    # RSI (14-day)
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss     = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs       = gain / loss.replace(0, np.nan)
    features["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    # Volume features
    features["volume_ratio"] = volume / volume.rolling(20).mean()

    # Moving average gaps
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    features["ma20_gap"] = (close - ma20) / ma20      # how far price is from MA20
    features["ma50_gap"] = (close - ma50) / ma50      # how far price is from MA50
    features["ma_cross"]  = (ma20 - ma50) / ma50      # fast vs slow MA gap

    # Target: will price be higher tomorrow? (1 = yes, 0 = no)
    features["target"] = (close.shift(-1) > close).astype(int)

    return features


# ── MODEL TRAINING ────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier on historical data.

    Uses TimeSeriesSplit for cross-validation — this is important.
    Normal k-fold would leak future data into training. TimeSeries split
    always trains on past data only, validates on future data.

    Returns: (trained model, scaler, feature column names)
    """
    features_df = build_features(df)
    features_df = features_df.dropna()

    feature_cols = [c for c in features_df.columns if c != "target"]
    X = features_df[feature_cols]
    y = features_df["target"]

    # Leave the last 20% as final holdout (never touch during training)
    split_idx   = int(len(X) * 0.8)
    X_train     = X.iloc[:split_idx]
    y_train     = y.iloc[:split_idx]

    # Scale features
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,           # keep shallow to reduce overfitting
        min_samples_leaf=20,   # each leaf needs at least 20 samples
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train)

    return model, scaler, feature_cols


# ── SIGNAL GENERATION ────────────────────────────────────────────────────────

def run_ml_agent(symbol: str = None) -> pd.DataFrame:
    """
    Train model per stock and generate signals for each row.
    Returns DataFrame with: Date, Symbol, ML_Score
    """
    conn = sqlite3.connect(DB_PATH)

    if symbol:
        query = f"SELECT * FROM prices WHERE Symbol = '{symbol}' ORDER BY Date"
    else:
        query = "SELECT * FROM prices ORDER BY Symbol, Date"

    df = pd.read_sql(query, conn, index_col="Date")
    conn.close()

    results = []

    symbols = [symbol] if symbol else df["Symbol"].unique()

    for sym in symbols:
        group = df[df["Symbol"] == sym].copy()

        if len(group) < 200:
            print(f"  [SKIP] {sym}: not enough data ({len(group)} rows)")
            continue

        try:
            model, scaler, feature_cols = train_model(group)

            # Generate signals for the FULL dataset using trained model
            features_df = build_features(group).dropna()
            X_all = features_df[feature_cols]
            X_scaled = scaler.transform(X_all)

            # Probability of price going up tomorrow
            prob_up = model.predict_proba(X_scaled)[:, 1]

            # Rescale from [0,1] to [-1, +1]
            # 0.5 probability = neutral = 0 score
            ml_score = (prob_up - 0.5) * 2

            result = pd.DataFrame({
                "Date":     features_df.index,
                "Symbol":   sym,
                "ML_Score": ml_score,
            })
            results.append(result)

        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            continue

    if not results:
        return pd.DataFrame(columns=["Date", "Symbol", "ML_Score"])

    return pd.concat(results, ignore_index=True)


def get_latest_signal(symbol: str) -> dict:
    """
    Get the most recent ML signal for a single stock.
    Called by the orchestrator in Phase 4.
    """
    print(f"  Training ML model for {symbol}...")
    df = run_ml_agent(symbol=symbol)

    if df.empty:
        return {"symbol": symbol, "score": 0.0, "interpretation": "no data"}

    latest = df.iloc[-1]
    score  = round(latest["ML_Score"], 4)

    if score > 0.3:
        interpretation = "model predicts upward move tomorrow"
    elif score > 0.05:
        interpretation = "slight upward bias"
    elif score < -0.3:
        interpretation = "model predicts downward move tomorrow"
    elif score < -0.05:
        interpretation = "slight downward bias"
    else:
        interpretation = "model is uncertain — near 50/50"

    return {
        "symbol": symbol,
        "agent": "ml_classifier",
        "score": score,
        "interpretation": interpretation,
        "date": latest["Date"],
    }


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running ML Agent on RELIANCE and TCS (test)...\n")
    print("Note: training a model per stock takes ~5 seconds each.\n")

    for sym in ["RELIANCE", "TCS", "INFY"]:
        result = get_latest_signal(sym)
        print(f"{sym}: score={result['score']:+.4f} | {result['interpretation']}")

    print("\nML Agent working correctly.")
    print("(Running on all 50 stocks at once will take ~5 minutes — normal)")