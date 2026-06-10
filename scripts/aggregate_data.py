#!/usr/bin/env python3
"""
XAUUSD Data Aggregation Script

Reads 1-minute XAUUSD data and aggregates into M15, H1, H4, D1 timeframes.
For each timeframe computes OHLCV (open, high, low, close, volume).
"""

import os
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_FILE = os.path.join(DATA_DIR, "XAUUSD_1m.csv")

OUTPUT_FILES = {
    "M15": os.path.join(DATA_DIR, "XAUUSD_M15.csv"),
    "H1":  os.path.join(DATA_DIR, "XAUUSD_H1.csv"),
    "H4":  os.path.join(DATA_DIR, "XAUUSD_H4.csv"),
    "D1":  os.path.join(DATA_DIR, "XAUUSD_D1.csv"),
}

# Resample rule mapping for pandas
RESAMPLE_RULES = {
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}

# OHLCV aggregation specification
AGG_SPEC = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}


def load_1m_data(filepath: str) -> pd.DataFrame:
    """Load 1-minute XAUUSD data from CSV."""
    print(f"[INFO] Loading 1-minute data from: {filepath}")
    df = pd.read_csv(
        filepath,
        dtype={"time": str, "open": float, "high": float, "low": float, "close": float, "volume": int},
    )
    # Parse time column manually — data has mixed formats:
    #   early rows: "2024-01-02 05:01:00"  (naive, space separator)
    #   later rows: "2025-11-17T14:02:00Z" (ISO8601 with Z suffix)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
    # Convert to tz-naive for clean resampling
    df["time"] = df["time"].dt.tz_localize(None)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    print(f"[INFO] Loaded {len(df):,} rows  |  "
          f"Date range: {df.index.min()} → {df.index.max()}")
    return df


def aggregate_timeframe(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to the given timeframe rule."""
    agg_df = df.resample(rule).agg(AGG_SPEC)
    # Drop rows where all values are NaN (periods with no data)
    agg_df.dropna(subset=["open", "close"], inplace=True)
    # Convert volume back to int (sum of ints)
    agg_df["volume"] = agg_df["volume"].astype(int)
    return agg_df


def print_summary(label: str, df: pd.DataFrame) -> None:
    """Print summary statistics for a timeframe."""
    print(f"\n{'='*60}")
    print(f"  {label} Summary")
    print(f"{'='*60}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  Price high: {df['high'].max():.2f}")
    print(f"  Price low : {df['low'].min():.2f}")
    print(f"  Price span: {df['high'].max() - df['low'].min():.2f}")
    print(f"  Avg volume: {df['volume'].mean():,.0f}")
    print(f"  Total vol : {df['volume'].sum():,}")


def main():
    # Load source data
    df_1m = load_1m_data(INPUT_FILE)

    results = {}

    for label, rule in RESAMPLE_RULES.items():
        print(f"\n[INFO] Aggregating to {label} (rule={rule})...")
        agg_df = aggregate_timeframe(df_1m, rule)

        # Save to CSV
        out_path = OUTPUT_FILES[label]
        agg_df.to_csv(out_path)
        print(f"[INFO] Saved {len(agg_df):,} rows → {out_path}")

        # Store for summary
        results[label] = agg_df

    # Print all summaries
    print("\n" + "=" * 60)
    print("  AGGREGATION COMPLETE — Summary Report")
    print("=" * 60)

    for label, agg_df in results.items():
        print_summary(label, agg_df)

    print(f"\n{'='*60}")
    print("  All files saved to:")
    for label, path in OUTPUT_FILES.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"    {label}: {path}  ({size_kb:.1f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
