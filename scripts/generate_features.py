#!/usr/bin/env python3
"""
generate_features.py — Generate XAUUSD feature CSVs using the 3-Concept Framework
=================================================================================
Reads M15 and H1 data, computes base + advanced features, saves results.
"""

import sys
import os
import time

# Add project scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from feature_engineering_v2 import compute_base_features, compute_advanced_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

M15_PATH = os.path.join(DATA_DIR, "XAUUSD_M15.csv")
H1_PATH = os.path.join(DATA_DIR, "XAUUSD_H1.csv")
OUT_BASE = os.path.join(DATA_DIR, "XAUUSD_M15_features_v2.csv")
OUT_ENHANCED = os.path.join(DATA_DIR, "XAUUSD_M15_features_v2_enhanced.csv")


def main():
    print("=" * 70)
    print("XAUUSD Feature Generation — 3-Concept Framework (Trend/Momentum/Cycle)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load M15 data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading M15 data from {M15_PATH}...")
    t0 = time.time()
    df_m15 = pd.read_csv(M15_PATH, parse_dates=["time"], index_col="time")
    df_m15 = df_m15.sort_index()
    print(f"      M15: {len(df_m15):,} rows | {df_m15.index[0]} → {df_m15.index[-1]}")
    print(f"      Columns: {list(df_m15.columns)}")
    print(f"      Loaded in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 2. Load H1 data
    # ------------------------------------------------------------------
    print(f"\n[2/5] Loading H1 data from {H1_PATH}...")
    t0 = time.time()
    df_h1 = pd.read_csv(H1_PATH, parse_dates=["time"], index_col="time")
    df_h1 = df_h1.sort_index()
    print(f"      H1:  {len(df_h1):,} rows | {df_h1.index[0]} → {df_h1.index[-1]}")
    print(f"      Loaded in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. Compute base features
    # ------------------------------------------------------------------
    print(f"\n[3/5] Computing base features (Trend/Momentum/Cycle)...")
    t0 = time.time()
    df_base = compute_base_features(df_m15, df_h1)
    elapsed = time.time() - t0
    feature_cols = [c for c in df_base.columns if c not in ["open", "high", "low", "close", "volume"]]
    print(f"      Base feature columns: {len(feature_cols)}")
    print(f"      Total columns (incl. OHLCV): {len(df_base.columns)}")
    print(f"      Rows: {len(df_base):,}")
    print(f"      Computed in {elapsed:.2f}s")

    # Breakdown by concept
    trend_cols = [c for c in feature_cols if c.startswith("trend_")]
    mom_cols = [c for c in feature_cols if c.startswith("mom_")]
    cyc_cols = [c for c in feature_cols if c.startswith("cyc_")]
    print(f"      CONCEPT 1 — TREND:    {len(trend_cols)} features")
    print(f"      CONCEPT 2 — MOMENTUM: {len(mom_cols)} features")
    print(f"      CONCEPT 3 — CYCLE:    {len(cyc_cols)} features")

    # NaN statistics
    nan_counts = df_base[feature_cols].isna().sum()
    valid_rows = df_base[feature_cols].dropna().shape[0]
    print(f"      Valid rows (no NaN in features): {valid_rows:,} / {len(df_base):,}")
    print(f"      NaN% per column (max): {nan_counts.max() / len(df_base) * 100:.1f}%")

    # Save base features
    print(f"\n      Saving base features to {OUT_BASE}...")
    df_base.to_csv(OUT_BASE)
    fsize_mb = os.path.getsize(OUT_BASE) / (1024 * 1024)
    print(f"      Saved: {fsize_mb:.2f} MB")

    # ------------------------------------------------------------------
    # 4. Compute advanced features
    # ------------------------------------------------------------------
    print(f"\n[4/5] Computing advanced features...")
    t0 = time.time()
    df_enhanced = compute_advanced_features(df_base)
    elapsed = time.time() - t0

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    all_feature_cols = [c for c in df_enhanced.columns if c not in ohlcv_cols]
    base_only = set(feature_cols)
    adv_only = set(all_feature_cols) - base_only
    print(f"      Advanced feature columns: {len(adv_only)}")
    print(f"      Total feature columns: {len(all_feature_cols)}")
    print(f"      Total columns (incl. OHLCV): {len(df_enhanced.columns)}")
    print(f"      Rows: {len(df_enhanced):,}")
    print(f"      Computed in {elapsed:.2f}s")

    # NaN statistics
    valid_rows_enh = df_enhanced[all_feature_cols].dropna().shape[0]
    print(f"      Valid rows (no NaN in features): {valid_rows_enh:,} / {len(df_enhanced):,}")

    # Sample of advanced feature names
    adv_list = sorted(adv_only)
    print(f"\n      Sample advanced features ({min(10, len(adv_list))}/{len(adv_list)}):")
    for name in adv_list[:10]:
        print(f"        • {name}")
    if len(adv_list) > 10:
        print(f"        ... and {len(adv_list) - 10} more")

    # Save enhanced features
    print(f"\n      Saving enhanced features to {OUT_ENHANCED}...")
    df_enhanced.to_csv(OUT_ENHANCED)
    fsize_mb = os.path.getsize(OUT_ENHANCED) / (1024 * 1024)
    print(f"      Saved: {fsize_mb:.2f} MB")

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    print(f"\n[5/5] Summary Statistics")
    print("-" * 50)

    # Feature statistics for key indicators (no NaN)
    df_valid = df_enhanced[all_feature_cols].dropna()
    print(f"\n  Valid dataset size: {len(df_valid):,} rows × {len(all_feature_cols)} features")

    # Select a few key features for stats
    key_features = [
        "trend_price_vs_sma20_atr",
        "mom_rsi14",
        "mom_macd_histogram",
        "cyc_bb_pct_b",
        "cyc_atr14_normalized",
        "trend_adx14",
    ]
    existing_keys = [f for f in key_features if f in df_valid.columns]
    if existing_keys:
        print(f"\n  Key feature statistics (valid rows only):")
        stats = df_valid[existing_keys].describe().T[["mean", "std", "min", "max"]]
        for feat, row in stats.iterrows():
            print(f"    {feat:40s}  mean={row['mean']:10.4f}  std={row['std']:10.4f}  "
                  f"min={row['min']:10.4f}  max={row['max']:10.4f}")

    # Target distribution preview (if we were to create it)
    if "close" in df_enhanced.columns:
        future_close = df_enhanced["close"].shift(-4)
        target = (future_close > df_enhanced["close"]).astype(int)
        valid_target = target.dropna()
        if len(valid_target) > 0:
            bull_pct = valid_target.mean() * 100
            print(f"\n  Target distribution (horizon=4, M15=60min ahead):")
            print(f"    Bullish (1): {bull_pct:.1f}%")
            print(f"    Bearish (0): {100 - bull_pct:.1f}%")

    print(f"\n{'=' * 70}")
    print(f"FEATURE GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Output files:")
    print(f"    Base:       {OUT_BASE}")
    print(f"    Enhanced:   {OUT_ENHANCED}")


if __name__ == "__main__":
    main()
