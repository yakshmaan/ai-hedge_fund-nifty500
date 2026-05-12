"""
clean_data.py
-------------
Reads raw CSVs from data/raw/, cleans them, saves to data/clean/.

What "cleaning" means for financial data:
  1. Remove rows where Close price is 0 or negative (bad data)
  2. Forward-fill missing dates (market holidays create gaps)
  3. Flag and remove extreme outliers (price jumps > 50% in one day)
  4. Ensure consistent date range across all stocks

Run AFTER fetch_data.py.

Usage:
    python pipeline/clean_data.py
"""

import pandas as pd
import numpy as np
import os
import glob

# ── CONFIG ────────────────────────────────────────────────────────────────────

RAW_DIR   = "data/raw"
CLEAN_DIR = "data/clean"

# If a stock's daily return is beyond this, it's probably bad data
# (real stocks occasionally do move 50%+ but it's rare enough to flag)
OUTLIER_THRESHOLD = 0.50  # 50% single-day move


# ── CLEANING LOGIC ────────────────────────────────────────────────────────────

def clean_stock(filepath: str) -> pd.DataFrame | None:
    """
    Load one raw CSV, clean it, return a cleaned DataFrame.
    Returns None if the file is too broken to save.
    """
    symbol = os.path.basename(filepath).replace(".csv", "")

    try:
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    except Exception as e:
        print(f"  [ERROR] Could not read {filepath}: {e}")
        return None

    original_len = len(df)

    # Step 1: Drop rows with 0 or negative Close (clearly bad data)
    df = df[df["Close"] > 0]

    # Step 2: Drop rows where Volume is 0 (stock wasn't traded — usually bad data)
    # Keep NaN volumes (some stocks don't report them) but drop explicit zeros
    df = df[(df["Volume"] != 0) | df["Volume"].isna()]

    # Step 3: Sort by date ascending (should already be, but be safe)
    df = df.sort_index()

    # Step 4: Forward-fill missing values caused by holidays/weekends
    # We reindex to a complete business day range so gaps are explicit
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
    df = df.reindex(full_range)
    df = df.ffill()  # forward fill: use previous day's price for missing days
    df.index.name = "Date"

    # Step 5: Flag extreme outliers using daily returns
    df["Daily_Return"] = df["Close"].pct_change()
    outliers = df["Daily_Return"].abs() > OUTLIER_THRESHOLD
    n_outliers = outliers.sum()

    if n_outliers > 0:
        print(f"  [WARN] {symbol}: {n_outliers} outlier days flagged (>50% move)")
        # We flag but don't remove — could be real (stock splits, circuit breakers)
        # In Phase 2 you'll handle this more carefully
        df["Is_Outlier"] = outliers

    else:
        df["Is_Outlier"] = False

    # Step 6: Final check — need at least 200 rows (roughly 1 trading year)
    if len(df) < 200:
        print(f"  [SKIP] {symbol}: only {len(df)} rows after cleaning — too short")
        return None

    rows_removed = original_len - len(df[df["Close"].notna()])
    return df


def clean_all():
    """
    Process all raw CSVs and save cleaned versions.
    """
    os.makedirs(CLEAN_DIR, exist_ok=True)

    raw_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    if not raw_files:
        print(f"No CSV files found in {RAW_DIR}/. Run fetch_data.py first.")
        return

    print(f"Cleaning {len(raw_files)} files...\n")

    success = 0
    skipped = []

    for i, filepath in enumerate(sorted(raw_files), 1):
        symbol = os.path.basename(filepath).replace(".csv", "")
        print(f"[{i:02d}/{len(raw_files)}] {symbol}", end=" ... ")

        df = clean_stock(filepath)

        if df is not None:
            out_path = os.path.join(CLEAN_DIR, f"{symbol}.csv")
            df.to_csv(out_path)
            print(f"clean ({len(df)} rows)")
            success += 1
        else:
            skipped.append(symbol)

    print(f"\n{'─'*50}")
    print(f"Done. {success}/{len(raw_files)} stocks cleaned successfully.")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print(f"Clean data saved to: {CLEAN_DIR}/")


if __name__ == "__main__":
    clean_all()