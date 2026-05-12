"""
store_data_nifty500.py
----------------------
Loads all cleaned CSVs into a single SQLite database: nifty500.db
 
Usage:
    python data/store_data_nifty500.py
"""
 
import pandas as pd
import sqlite3
import os
import glob
 
CLEAN_DIR = "data/clean"
DB_PATH   = "data/nifty500.db"
 
 
def build_database():
    clean_files = glob.glob(os.path.join(CLEAN_DIR, "*.csv"))
    if not clean_files:
        print(f"No files found in {CLEAN_DIR}/. Run clean_data.py first.")
        return
 
    print(f"Loading {len(clean_files)} stocks into database...\n")
    all_dfs = []
 
    for filepath in sorted(clean_files):
        symbol = os.path.basename(filepath).replace(".csv", "")
        df     = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        df["Symbol"] = symbol
        all_dfs.append(df)
        print(f"  Loaded {symbol} ({len(df)} rows)")
 
    combined        = pd.concat(all_dfs).reset_index()
    combined["Date"] = combined["Date"].astype(str)
 
    conn = sqlite3.connect(DB_PATH)
    combined.to_sql("prices", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON prices (Symbol, Date)")
    conn.commit()
 
    row_count    = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    symbol_count = conn.execute("SELECT COUNT(DISTINCT Symbol) FROM prices").fetchone()[0]
    date_range   = conn.execute("SELECT MIN(Date), MAX(Date) FROM prices").fetchone()
    conn.close()
 
    print(f"\n{'─'*50}")
    print(f"Database: {DB_PATH}")
    print(f"  Stocks : {symbol_count}")
    print(f"  Rows   : {row_count:,}")
    print(f"  From   : {date_range[0]}")
    print(f"  To     : {date_range[1]}")
 
 
if __name__ == "__main__":
    build_database()
 