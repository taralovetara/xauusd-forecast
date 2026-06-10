#!/usr/bin/env python3
"""
h1_pipeline.py — XAUUSD H1 Prediction Pipeline
================================================

The core script that ties together:
  1. Technical analysis (indicators + signals + confluence)
  2. ML prediction (stacked ensemble)
  3. Combined confidence scoring
  4. Entry/SL/TP generation
  5. Memory persistence
  6. Formatted output

Pipeline Steps
--------------
1. Load M15/H1 data from CSV files
2. Run technical analysis
3. Run ML prediction using trained models
4. Combine technical + ML analysis
5. Generate entry/SL/TP
6. Save prediction to memory/predictions.json
7. Print formatted prediction output
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow imports from parent scripts/ directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SCRIPTS_DIR)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, BASE_DIR)

from technical_analysis import (
    calculate_indicators,
    calculate_confluence_score,
    get_session_context,
    get_technical_signals,
    calculate_gold_stop_loss,
    enrich_signals_with_volume,
    enrich_signals_with_sr,
)
from feature_engineering_v2 import compute_base_features, compute_advanced_features
from memory_manager import load_predictions, save_predictions, add_prediction, validate_prediction
from oanda_data import OandaDataFetcher, refresh_with_yfinance_fallback
from twelvedata_fetcher import TwelveDataFetcher, refresh_live_data_smart

import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")

M15_CSV = os.path.join(DATA_DIR, "XAUUSD_M15.csv")
H1_CSV = os.path.join(DATA_DIR, "XAUUSD_H1.csv")
PREDICTIONS_JSON = os.path.join(MEMORY_DIR, "predictions.json")

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_v2_model.json")
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_v2_model.txt")
CB_MODEL_PATH = os.path.join(MODEL_DIR, "cb_v2_model.cbm")
META_LEARNER_PATH = os.path.join(MODEL_DIR, "meta_learner_v2.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_v2.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list_v2.json")


# ---------------------------------------------------------------------------
# Confidence label helper
# ---------------------------------------------------------------------------

def _confidence_label(confidence: float) -> str:
    """Convert 0-10 confidence score to label."""
    if confidence >= 7.0:
        return "HIGH"
    elif confidence >= 4.0:
        return "MEDIUM"
    else:
        return "LOW"


def _ml_confidence_label(ml_conf: float) -> str:
    """Convert ML probability confidence to label."""
    # ml_conf is max(prob_bull, prob_bear) from meta-learner
    if ml_conf >= 0.70:
        return "HIGH"
    elif ml_conf >= 0.58:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

def refresh_live_data():
    """Refresh data from OANDA → Twelve Data → yfinance (fallback chain)."""
    print("[0/7] Refreshing live data...")

    try:
        result = refresh_live_data_smart()
        data_source = result.get("source", "unknown")
        print(f"      Data source: {data_source}")
        return data_source
    except Exception as e:
        print(f"      All data sources failed: {e}")
        data_source = "csv (stale)"
        print(f"      Data source: {data_source}")
        return data_source


def load_data():
    """Load M15 and H1 CSV data."""
    print("[1/7] Loading data...")

    df_m15 = pd.read_csv(
        M15_CSV,
        index_col=0,
        parse_dates=True,
        date_format="ISO8601",
    )
    if hasattr(df_m15.index, 'tz') and df_m15.index.tz is not None:
        df_m15.index = df_m15.index.tz_localize(None)

    df_h1 = pd.read_csv(
        H1_CSV,
        index_col=0,
        parse_dates=True,
        date_format="ISO8601",
    )
    if hasattr(df_h1.index, 'tz') and df_h1.index.tz is not None:
        df_h1.index = df_h1.index.tz_localize(None)

    # Ensure column names are lowercase
    df_m15.columns = [c.lower() for c in df_m15.columns]
    df_h1.columns = [c.lower() for c in df_h1.columns]

    print(f"      M15: {len(df_m15):,} rows | Last: {df_m15.index[-1]}")
    print(f"      H1:  {len(df_h1):,} rows | Last: {df_h1.index[-1]}")

    return df_m15, df_h1


# ---------------------------------------------------------------------------
# Step 2: Technical analysis
# ---------------------------------------------------------------------------

def run_technical_analysis(df_m15, df_h1):
    """
    Run full technical analysis pipeline on H1 data.

    Returns: (df_h1_with_indicators, signals, tech_direction, tech_confidence, session_ctx)
    """
    print("[2/7] Running technical analysis...")

    # Calculate indicators on H1 data
    df_h1_ind = calculate_indicators(df_h1, timeframe="H1")

    # Get signals for latest bar
    signals = get_technical_signals(df_h1_ind)

    # Enrich with volume and S/R
    signals = enrich_signals_with_volume(df_h1_ind, signals)
    signals = enrich_signals_with_sr(df_h1_ind, signals)

    # Determine technical direction
    # Use EMA cross + MACD + trend alignment
    ema_dir = signals.get("ema_cross", "NEUTRAL")
    macd_dir = signals.get("macd_signal", "NEUTRAL")
    trend = signals.get("trend", "SIDEWAYS")

    # Count bullish vs bearish signals
    bull_count = 0
    bear_count = 0
    for key in ["ema_cross", "macd_signal"]:
        if signals.get(key) == "BULLISH":
            bull_count += 1
        elif signals.get(key) == "BEARISH":
            bear_count += 1

    # RSI direction hint
    rsi_val = signals.get("rsi_value", 50)
    if rsi_val is not None:
        if rsi_val < 40:
            bull_count += 0.5
        elif rsi_val > 60:
            bear_count += 0.5

    # Stochastic hint
    stoch = signals.get("stoch_signal", "NEUTRAL")
    if stoch == "OVERSOLD":
        bull_count += 0.5
    elif stoch == "OVERBOUGHT":
        bear_count += 0.5

    if bull_count > bear_count:
        tech_direction = "BULLISH"
    elif bear_count > bull_count:
        tech_direction = "BEARISH"
    else:
        # Tie-breaker: use trend
        if trend == "UPTREND":
            tech_direction = "BULLISH"
        elif trend == "DOWNTREND":
            tech_direction = "BEARISH"
        else:
            tech_direction = "BULLISH"  # Default

    # Calculate confluence score for that direction
    tech_confidence = calculate_confluence_score(signals, tech_direction)

    # Session context
    latest_ts = df_h1.index[-1]
    session_ctx = get_session_context(latest_ts)

    print(f"      Direction: {tech_direction}")
    print(f"      Confluence: {tech_confidence:.1f}/10")
    print(f"      Session: {session_ctx.get('current_session', 'N/A')}")

    return df_h1_ind, signals, tech_direction, tech_confidence, session_ctx


# ---------------------------------------------------------------------------
# Step 3: ML prediction
# ---------------------------------------------------------------------------

def run_ml_prediction(df_m15, df_h1):
    """
    Run ML prediction using the stacked ensemble model.

    Returns: (ml_direction, ml_confidence, ml_probability, top_features)
    """
    print("[3/7] Running ML prediction...")

    # Load model artifacts
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)

    # Load LightGBM — use the Booster directly for predict.
    # The saved model is a text file from booster_.save_model(), so we
    # load it directly as a Booster object and use predict() for probability.
    lgbm_booster = lgb.Booster(model_file=LGBM_MODEL_PATH)

    cb_model = CatBoostClassifier()
    cb_model.load_model(CB_MODEL_PATH)

    meta_model = joblib.load(META_LEARNER_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_LIST_PATH, "r") as f:
        feature_data = json.load(f)
    feature_names = feature_data["feature_names"]

    print(f"      Loaded {len(feature_names)} features, scaler, and 4 models")

    # Compute features for the latest bar
    # Need a sufficient window for indicator warm-up
    # Use last 500 M15 bars for feature computation
    window = min(500, len(df_m15))
    df_m15_window = df_m15.iloc[-window:].copy()

    # Compute base features
    df_features = compute_base_features(df_m15_window, df_h1)

    # Compute advanced features
    df_enhanced = compute_advanced_features(df_features)

    # Get the last row (most recent bar)
    last_row = df_enhanced.iloc[-1]

    # Extract feature columns that match feature_list
    available_features = [f for f in feature_names if f in df_enhanced.columns]
    missing_features = [f for f in feature_names if f not in df_enhanced.columns]

    if missing_features:
        # Fill missing features with 0
        print(f"      Warning: {len(missing_features)} features missing, filling with 0")

    # Build feature vector
    feature_vector = []
    for fname in feature_names:
        if fname in df_enhanced.columns:
            val = last_row[fname]
            if pd.isna(val) or np.isinf(val):
                feature_vector.append(0.0)
            else:
                feature_vector.append(float(val))
        else:
            feature_vector.append(0.0)

    X = np.array([feature_vector], dtype=np.float32)

    # Scale features
    X_scaled = scaler.transform(X)

    # Get probabilities from base models
    # XGBoost
    xgb_proba = xgb_model.predict_proba(X_scaled)[0, 1]

    # LightGBM — use booster predict directly (returns raw probability)
    lgbm_proba = float(lgbm_booster.predict(X_scaled)[0])

    # CatBoost
    cb_proba = cb_model.predict_proba(X_scaled)[0, 1]

    print(f"      Base model probabilities: XGB={xgb_proba:.3f} LGBM={lgbm_proba:.3f} CB={cb_proba:.3f}")

    # Stack meta-features
    meta_X = np.array([[xgb_proba, lgbm_proba, cb_proba]])

    # Meta-learner prediction
    ensemble_proba = meta_model.predict_proba(meta_X)[0, 1]

    print(f"      Ensemble probability: {ensemble_proba:.3f}")

    # Determine direction and confidence
    if ensemble_proba >= 0.5:
        ml_direction = "BULLISH"
        ml_probability = ensemble_proba
    else:
        ml_direction = "BEARISH"
        ml_probability = 1.0 - ensemble_proba

    # ML confidence: scale probability to 0-1 range
    # Distance from 0.5, doubled to [0,1]
    ml_confidence = abs(ensemble_proba - 0.5) * 2

    # Get top 3 important features from the model (use raw feature values for display)
    top_features = _get_top_features(xgb_model, feature_names, X_scaled, X_raw=X, n=3)

    print(f"      ML Direction: {ml_direction}")
    print(f"      ML Confidence: {ml_confidence:.3f}")

    return ml_direction, ml_confidence, ml_probability, top_features


def _get_top_features(model, feature_names, X, X_raw=None, n=3):
    """
    Get top n features by importance-weighted contribution.

    Uses scaled features for contribution calculation but raw (unscaled)
    feature values for display to the user.
    """
    try:
        importances = model.feature_importances_
        # Weight by absolute feature value (scaled)
        contributions = np.abs(X[0]) * importances
        top_idx = np.argsort(contributions)[::-1][:n]

        # Use raw values for display if available
        display_values = X_raw[0] if X_raw is not None else X[0]

        top_features = []
        for idx in top_idx:
            top_features.append({
                "name": feature_names[idx],
                "importance": float(importances[idx]),
                "value": float(display_values[idx]),
                "contribution": float(contributions[idx]),
            })
        return top_features
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Step 4: Combine technical + ML
# ---------------------------------------------------------------------------

def combine_analysis(tech_direction, tech_confidence, ml_direction, ml_confidence):
    """
    Combine technical and ML analysis.

    Rules:
    - If both agree: boost confidence by 1.0
    - If they disagree: reduce confidence by 1.5
    - Final direction follows the higher-confidence analysis
    """
    print("[4/7] Combining technical + ML analysis...")

    if tech_direction == ml_direction:
        combined_direction = tech_direction
        # Boost confidence
        # Scale ML confidence to 0-10 range
        ml_contrib = ml_confidence * 3.0  # Scale 0-1 to 0-3
        combined_confidence = tech_confidence + ml_contrib + 1.0  # Agreement bonus
        agreement = True
    else:
        # Disagreement: use the analysis with higher confidence
        ml_contrib = ml_confidence * 3.0
        if tech_confidence >= ml_contrib:
            combined_direction = tech_direction
        else:
            combined_direction = ml_direction

        combined_confidence = max(tech_confidence, ml_contrib) - 1.5  # Disagreement penalty
        agreement = False

    # Clamp to [1, 10]
    combined_confidence = max(1.0, min(10.0, combined_confidence))

    label = _confidence_label(combined_confidence)
    print(f"      Tech: {tech_direction} ({tech_confidence:.1f}/10)")
    print(f"      ML:   {ml_direction} ({ml_confidence:.3f})")
    print(f"      Combined: {combined_direction} ({combined_confidence:.1f}/10, {label})")
    print(f"      Agreement: {'YES' if agreement else 'NO'}")

    return combined_direction, combined_confidence, label, agreement


# ---------------------------------------------------------------------------
# Step 5: Generate entry/SL/TP
# ---------------------------------------------------------------------------

def generate_levels(direction, confidence, signals, df_h1_ind):
    """Generate entry price, stop loss, and take profit levels."""
    print("[5/7] Generating entry/SL/TP levels...")

    price = signals["price"]
    atr = signals.get("atr_value", 20.0)

    # Entry is the current close
    entry_price = price

    # Calculate SL/TP using gold-specific method
    stops = calculate_gold_stop_loss(atr, direction, entry_price, multiplier=1.5)

    # Adjust TP2 based on confidence (higher confidence = aim for TP2/TP3)
    # Use TP1 as the primary target, TP2 as secondary
    stop_loss = stops["stop_loss"]
    take_profit_1 = stops["take_profit_1"]  # 1:1 R:R
    take_profit_2 = stops["take_profit_2"]  # 1:2 R:R

    stop_distance = stops["stop_distance"]

    # Risk:Reward ratio
    rr_ratio = (take_profit_2 - entry_price) / stop_distance if direction == "BULLISH" else (entry_price - take_profit_2) / stop_distance

    print(f"      Entry: ${entry_price:,.2f}")
    print(f"      SL: ${stop_loss:,.2f} ({stop_distance:.2f} pts, {1.5}x ATR)")
    print(f"      TP1: ${take_profit_1:,.2f}")
    print(f"      TP2: ${take_profit_2:,.2f}")
    print(f"      R:R = 1:{rr_ratio:.1f}")

    return {
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit_1": round(take_profit_1, 2),
        "take_profit_2": round(take_profit_2, 2),
        "stop_distance": round(stop_distance, 2),
        "rr_ratio": round(rr_ratio, 1),
        "atr_value": round(atr, 2),
    }


# ---------------------------------------------------------------------------
# Step 6: Build confluences and risks lists
# ---------------------------------------------------------------------------

def build_confluences_and_risks(signals, ml_direction, agreement, session_ctx):
    """Build confluence and risk lists from signals."""
    confluences = []
    risks = []

    # EMA cross
    if signals.get("ema_cross") in ("BULLISH", "BEARISH"):
        ema_dir = signals["ema_cross"]
        confluences.append(f"EMA9/21 {ema_dir.lower()} crossover")

    # Trend alignment
    trend = signals.get("trend", "SIDEWAYS")
    if trend in ("UPTREND", "DOWNTREND"):
        confluences.append(f"H1 trend alignment ({trend.lower()})")

    # Stochastic
    stoch = signals.get("stoch_signal", "NEUTRAL")
    if stoch == "OVERSOLD":
        confluences.append("Stochastic oversold bounce")
    elif stoch == "OVERBOUGHT":
        confluences.append("Stochastic overbought reversal")

    # RSI
    rsi_val = signals.get("rsi_value")
    if rsi_val is not None:
        if rsi_val < 35:
            confluences.append("RSI recovering from oversold")
        elif rsi_val > 65:
            confluences.append("RSI weakening from overbought")

    # RSI divergence
    if signals.get("rsi_divergence") != "NONE":
        confluences.append(f"RSI {signals['rsi_divergence'].lower()} divergence")

    # MACD divergence
    if signals.get("macd_divergence") != "NONE":
        confluences.append(f"MACD {signals['macd_divergence'].lower()} divergence")

    # BB
    bb_signal = signals.get("bb_signal", "NEUTRAL")
    if bb_signal in ("OVERSOLD", "OVERBOUGHT"):
        confluences.append(f"BB {bb_signal.lower()} — mean reversion setup")

    # ML agreement
    if agreement:
        confluences.append("ML model agrees with technicals")
    else:
        risks.append("ML model disagrees with technicals")

    # Fibonacci
    price = signals.get("price")
    fib_levels = signals.get("fib_levels", {})
    atr = signals.get("atr_value", 20)
    if price and fib_levels and atr:
        for fib_name, fib_price in fib_levels.items():
            if any(lvl in fib_name for lvl in ["38.2%", "50.0%", "61.8%"]):
                if abs(price - fib_price) <= atr * 0.5:
                    confluences.append(f"Near key Fibonacci {fib_name}")
                    break

    # Volume
    if signals.get("volume_increasing"):
        confluences.append("Volume above average — conviction")
    else:
        risks.append("Below-average volume")

    # Session risks
    session = session_ctx.get("current_session", "")
    if "ASIAN" in session:
        risks.append("Low-liquidity Asian session")
    if session_ctx.get("is_london_fix"):
        risks.append("London Fix volatility")
    if session_ctx.get("is_comex_open"):
        confluences.append("COMEX open — high liquidity")

    # Next session ending
    next_session = session_ctx.get("next_session", "")
    time_to_next = session_ctx.get("time_to_next_session", "")
    if "OFF" in next_session:
        risks.append(f"Session ending soon ({time_to_next})")

    # S/R
    if signals.get("sr_nearby"):
        confluences.append("Near key S/R level")
    else:
        risks.append("No nearby S/R confirmation")

    # Limit to reasonable size
    confluences = confluences[:6]
    risks = risks[:4]

    # Ensure at least 1 risk
    if not risks:
        risks.append("General market risk")

    return confluences, risks


# ---------------------------------------------------------------------------
# Step 7: Format and print prediction
# ---------------------------------------------------------------------------

def format_prediction_output(
    direction, confidence, label, levels, signals, ml_direction,
    ml_confidence, ml_probability, top_features, confluences, risks, session_ctx, timestamp
):
    """Format and print the prediction output."""
    # Direction symbol
    arrow = "▲" if direction == "BULLISH" else "▼"

    # Confidence bar (0-10, 10 chars wide)
    filled = int(confidence)
    empty = 10 - filled
    conf_bar = "█" * filled + "░" * empty

    # ATR description
    atr = signals.get("atr_value", 0)
    if atr < 15:
        atr_desc = "low volatility"
    elif atr < 25:
        atr_desc = "moderate volatility"
    elif atr < 40:
        atr_desc = "high volatility"
    else:
        atr_desc = "extreme volatility"

    # RSI description
    rsi_val = signals.get("rsi_value", 50)
    rsi_sig = signals.get("rsi_signal", "NEUTRAL")
    if rsi_sig == "OVERSOLD":
        rsi_desc = "OVERSOLD, recovering"
    elif rsi_sig == "OVERBOUGHT":
        rsi_desc = "OVERBOUGHT, weakening"
    else:
        rsi_desc = "NEUTRAL"

    # MACD description
    macd_sig = signals.get("macd_signal", "NEUTRAL")
    if macd_sig == "BULLISH":
        macd_desc = "BULLISH crossover forming"
    elif macd_sig == "BEARISH":
        macd_desc = "BEARISH crossover forming"
    else:
        macd_desc = "NEUTRAL"

    # Stochastic description
    stoch_sig = signals.get("stoch_signal", "NEUTRAL")
    stoch_k = signals.get("stoch_k")
    # We don't have stoch_k directly from signals; use stoch_signal
    if stoch_sig == "OVERSOLD":
        stoch_desc = "OVERSOLD (bounce expected)"
    elif stoch_sig == "OVERBOUGHT":
        stoch_desc = "OVERBOUGHT (reversal expected)"
    else:
        stoch_desc = "NEUTRAL"

    # BB description
    bb_sig = signals.get("bb_signal", "NEUTRAL")
    bb_pct = signals.get("bb_position", 0.5)
    if bb_sig != "NEUTRAL":
        bb_desc = f"{bb_sig} (%B={bb_pct:.2f})"
    elif abs(bb_pct - 0.5) < 0.15:
        bb_desc = "Mid-range (no squeeze)"
    elif bb_pct < 0.2:
        bb_desc = "Near lower band"
    elif bb_pct > 0.8:
        bb_desc = "Near upper band"
    else:
        bb_desc = f"Mid-range (%B={bb_pct:.2f})"

    # Trend description
    trend = signals.get("trend", "SIDEWAYS")
    sma20 = signals.get("sma_20", 0)
    sma50 = signals.get("sma_50", 0)
    sma200 = signals.get("sma_200", 0)
    if trend == "UPTREND":
        trend_desc = f"UPTREND (SMA20>50>200)"
    elif trend == "DOWNTREND":
        trend_desc = f"DOWNTREND (SMA20<50<200)"
    else:
        trend_desc = "SIDEWAYS"

    # SL distance in points and ATR multiplier
    stop_dist = levels["stop_distance"]
    atr_val = levels["atr_value"]
    atr_mult = stop_dist / atr_val if atr_val > 0 else 1.5

    # TP distances
    entry = levels["entry_price"]
    tp1_dist = levels["take_profit_1"] - entry if direction == "BULLISH" else entry - levels["take_profit_1"]
    tp2_dist = levels["take_profit_2"] - entry if direction == "BULLISH" else entry - levels["take_profit_2"]

    # Format timestamp
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M") + " UTC"

    # Session
    session_name = session_ctx.get("current_session", "N/A")

    output = f"""
═══════════════════════════════════════════════════
  XAUUSD H1 FORECAST — {ts_str}
═══════════════════════════════════════════════════
  Direction:  {arrow} {direction}
  Confidence: {conf_bar} {confidence:.1f}/10 ({label})
  
  Entry:      ${entry:,.2f}
  Stop Loss:  ${levels['stop_loss']:,.2f}  (-{stop_dist:.2f} / {atr_mult:.1f}x ATR)
  TP 1:       ${levels['take_profit_1']:,.2f}  (+{tp1_dist:.2f})
  TP 2:       ${levels['take_profit_2']:,.2f}  (+{tp2_dist:.2f})
  Risk:Reward 1:{levels['rr_ratio']:.1f}
  
  ── Technical Analysis ──
  Trend:     {trend_desc}
  RSI(14):   {rsi_val:.1f} ({rsi_desc})
  MACD:      {macd_desc}
  Stochastic: {stoch_desc}
  ATR:       {atr:.2f} ({atr_desc})
  BB:        {bb_desc}
  
  ── ML Prediction ──
  Direction:  {ml_direction} ({ml_probability*100:.1f}% probability)
  Confidence: {ml_confidence:.2f} ({_ml_confidence_label(ml_confidence)})
"""

    if top_features:
        output += "  Top 3 features:\n"
        for i, feat in enumerate(top_features, 1):
            sign = "+" if feat.get("value", 0) >= 0 else ""
            output += f"    {i}. {feat['name']}: {sign}{feat.get('value', 0):.2f}\n"

    output += "\n  ── Confluences ──\n"
    for c in confluences:
        output += f"  ✓ {c}\n"

    output += "\n  ── Risks ──\n"
    for r in risks:
        output += f"  ⚠ {r}\n"

    output += f"\n  Session: {session_name}\n"
    output += "═══════════════════════════════════════════════════"

    return output


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full H1 prediction pipeline."""
    print("=" * 60)
    print("XAUUSD H1 PREDICTION PIPELINE")
    print("=" * 60)
    total_t0 = time.time()

    # Step 0: Refresh live data (OANDA or yfinance)
    data_source = refresh_live_data()

    # Step 1: Load data
    df_m15, df_h1 = load_data()

    # Step 2: Technical analysis
    df_h1_ind, signals, tech_direction, tech_confidence, session_ctx = run_technical_analysis(df_m15, df_h1)

    # Step 3: ML prediction
    try:
        ml_direction, ml_confidence, ml_probability, top_features = run_ml_prediction(df_m15, df_h1)
    except Exception as e:
        print(f"      ML prediction failed: {e}")
        print(f"      Falling back to technical-only prediction")
        ml_direction = tech_direction
        ml_confidence = 0.5
        ml_probability = 0.5
        top_features = []

    # Step 4: Combine analysis
    combined_direction, combined_confidence, combined_label, agreement = combine_analysis(
        tech_direction, tech_confidence, ml_direction, ml_confidence
    )

    # Step 5: Generate entry/SL/TP
    levels = generate_levels(combined_direction, combined_confidence, signals, df_h1_ind)

    # Step 6: Build confluences and risks
    confluences, risks = build_confluences_and_risks(
        signals, ml_direction, agreement, session_ctx
    )

    # Get the timestamp for the prediction
    timestamp = df_h1.index[-1]
    # Use the next H1 bar as the prediction bar
    import pandas as pd
    next_bar_ts = timestamp + pd.Timedelta(hours=1)

    # Step 7: Build prediction dict and save
    print("[6/7] Saving prediction to memory...")

    prediction_dict = {
        "id": "",  # Auto-generated
        "timestamp": next_bar_ts.isoformat() + "Z",
        "instrument": "XAUUSD",
        "timeframe": "H1",
        "direction": combined_direction,
        "confidence": round(combined_confidence, 1),
        "confidence_label": combined_label,
        "entry_price": levels["entry_price"],
        "stop_loss": levels["stop_loss"],
        "take_profit_1": levels["take_profit_1"],
        "take_profit_2": levels["take_profit_2"],
        "indicators": {
            "rsi": round(signals.get("rsi_value", 0), 1),
            "macd_signal": signals.get("macd_signal", "NEUTRAL"),
            "macd_divergence": signals.get("macd_divergence", "NONE"),
            "rsi_divergence": signals.get("rsi_divergence", "NONE"),
            "bb_signal": signals.get("bb_signal", "NEUTRAL"),
            "bb_position": round(signals.get("bb_position", 0.5), 3),
            "stoch_signal": signals.get("stoch_signal", "NEUTRAL"),
            "trend": signals.get("trend", "SIDEWAYS"),
            "atr": round(signals.get("atr_value", 0), 2),
            "ema_cross": signals.get("ema_cross", "NEUTRAL"),
        },
        "confluences": confluences,
        "risks": risks,
        "ml_direction": ml_direction,
        "ml_confidence": round(ml_confidence, 3),
        "session": session_ctx.get("current_session", "N/A"),
        "outcome": None,
        "outcome_price": None,
        "outcome_time": None,
        "pnl_pips": None,
    }

    # Load existing predictions, add, and save
    predictions = load_predictions(PREDICTIONS_JSON)
    predictions = add_prediction(predictions, prediction_dict)
    save_predictions(predictions, PREDICTIONS_JSON)
    print(f"      Saved prediction (ID: {predictions[-1]['id']})")
    print(f"      Total predictions: {len(predictions)}")

    # Step 8: Print formatted output
    print("[7/7] Generating forecast output...")
    print(f"      Data source: {data_source}")

    output = format_prediction_output(
        direction=combined_direction,
        confidence=combined_confidence,
        label=combined_label,
        levels=levels,
        signals=signals,
        ml_direction=ml_direction,
        ml_confidence=ml_confidence,
        ml_probability=ml_probability,
        top_features=top_features,
        confluences=confluences,
        risks=risks,
        session_ctx=session_ctx,
        timestamp=next_bar_ts,
    )

    print(output)

    # Step 9: Print clear trade signal
    from trade_signal import generate_trade_signal, generate_compact_signal
    trade_signal = generate_trade_signal(
        prediction_dict,
        ml_direction=ml_direction,
        ml_confidence=ml_confidence,
        agreement=agreement,
    )
    print(trade_signal)

    total_time = time.time() - total_t0
    print(f"\nPipeline completed in {total_time:.1f}s")

    # Store agreement in prediction dict for later use
    prediction_dict["agreement"] = agreement

    return prediction_dict


if __name__ == "__main__":
    main()
