#!/usr/bin/env python3
"""
xauusd_30min_forecast.py — XAUUSD Next 30-Min Forecast
=======================================================
Combines:
  1. TwelveData real-time price
  2. Technical Analysis (indicators + confluence score)
  3. ML Stacked Ensemble prediction (XGBoost + LightGBM + CatBoost → LR meta)
  4. Final signal: ML 40% + Technical 60%
"""

import json
import os
import sys
import joblib
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
# This file is in scripts/live/, so project root is 2 levels up
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # project root
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

sys.path.insert(0, SCRIPTS_DIR)

# ── Imports ────────────────────────────────────────────────────────────────
from technical_analysis import (
    calculate_indicators, get_technical_signals,
    enrich_signals_with_volume, enrich_signals_with_sr,
    calculate_confluence_score, get_session_context,
    calculate_gold_stop_loss,
)

API_KEY = "c2f7813446f540fb9b6cd83e5072b2a6"
SYMBOL = "XAU/USD"
BASE_URL = "https://api.twelvedata.com"


# ── 1. Fetch Real-Time Data ──────────────────────────────────────────────

def fetch_candles(interval="15min", outputsize=500):
    params = {"symbol": SYMBOL, "interval": interval, "outputsize": outputsize, "timezone": "UTC", "apikey": API_KEY}
    resp = requests.get(f"{BASE_URL}/time_series", params=params, timeout=30)
    data = resp.json()
    if "values" not in data:
        raise ValueError(f"TwelveData error: {data.get('message', data)}")
    records = []
    for c in data["values"]:
        records.append({
            "datetime": pd.to_datetime(c["datetime"], utc=True).tz_localize(None),
            "open": float(c["open"]), "high": float(c["high"]),
            "low": float(c["low"]), "close": float(c["close"]),
            "volume": int(c.get("volume", 0) or 0),
        })
    df = pd.DataFrame(records)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_current_price():
    params = {"symbol": SYMBOL, "apikey": API_KEY}
    resp = requests.get(f"{BASE_URL}/price", params=params, timeout=30)
    data = resp.json()
    return float(data["price"])


# ── 2. Technical Analysis ────────────────────────────────────────────────

def compute_technical_analysis(df):
    df = calculate_indicators(df, timeframe="M15")
    signals = get_technical_signals(df)
    signals = enrich_signals_with_volume(df, signals)
    signals = enrich_signals_with_sr(df, signals)
    if isinstance(df.index[-1], pd.Timestamp):
        signals["session"] = get_session_context(df.index[-1])
    return signals, df


# ── 3. ML Prediction ──────────────────────────────────────────────────────

def compute_ml_prediction(df_m15, df_h1=None):
    from feature_engineering_v2 import compute_base_features, compute_advanced_features
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    # Load models and config
    with open(os.path.join(MODELS_DIR, "feature_list_v2.json")) as f:
        fl_data = json.load(f)
    # feature_list_v2.json is a dict with 'feature_names' key
    if isinstance(fl_data, dict) and "feature_names" in fl_data:
        feature_list = fl_data["feature_names"]
    else:
        feature_list = fl_data  # backward compat if it's already a list
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_v2.pkl"))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgb_v2_model.json"))
    lgbm_model = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lgbm_v2_model.txt"))
    cb_model = CatBoostClassifier()
    cb_model.load_model(os.path.join(MODELS_DIR, "cb_v2_model.cbm"))
    meta_learner = joblib.load(os.path.join(MODELS_DIR, "meta_learner_v2.pkl"))

    # Compute features
    features = compute_base_features(df_m15, df_h1)
    features = compute_advanced_features(features)
    last_row = features.iloc[-1:]

    # Align to feature list
    for c in feature_list:
        if c not in last_row.columns:
            last_row[c] = 0.0
    X = last_row[feature_list].values.astype(np.float64)
    X_scaled = scaler.transform(X)

    # Base model predictions
    xgb_prob = float(xgb_model.predict_proba(X_scaled)[0][1])
    lgbm_prob = float(lgbm_model.predict(X_scaled)[0])
    cb_prob = float(cb_model.predict_proba(X_scaled)[0][1])

    # Meta-learner
    meta_features = np.array([[xgb_prob, lgbm_prob, cb_prob]])
    ml_prob = float(meta_learner.predict_proba(meta_features)[0][1])

    ml_direction = "BULLISH" if ml_prob > 0.5 else "BEARISH"
    ml_confidence = abs(ml_prob - 0.5) * 2

    return {
        "ml_prob": round(ml_prob, 4),
        "ml_direction": ml_direction,
        "ml_confidence": round(ml_confidence, 4),
        "xgb_prob": round(xgb_prob, 4),
        "lgbm_prob": round(lgbm_prob, 4),
        "cb_prob": round(cb_prob, 4),
        "features_available": sum(1 for c in feature_list if c in features.columns),
        "features_total": len(feature_list),
    }


# ── 4. Combine Signals ────────────────────────────────────────────────────

def combine_signals(ml_result, signals, current_price):
    ta_bull = calculate_confluence_score(signals, "BULLISH")
    ta_bear = calculate_confluence_score(signals, "BEARISH")

    if ml_result is None:
        direction = "BULLISH" if ta_bull > ta_bear else "BEARISH"
        ta_conf = (ta_bull if direction == "BULLISH" else ta_bear) / 10.0
        final_conf = ta_conf
    else:
        ta_direction = "BULLISH" if ta_bull > ta_bear else "BEARISH"
        ta_conf = (ta_bull if ta_direction == "BULLISH" else ta_bear) / 10.0
        ml_conf = ml_result["ml_confidence"]

        if ta_direction == ml_result["ml_direction"]:
            final_conf = ta_conf * 0.6 + ml_conf * 0.4
            direction = ta_direction
        else:
            final_conf = abs(ta_conf * 0.6 - ml_conf * 0.4)
            direction = ta_direction if ta_conf * 0.6 >= ml_conf * 0.4 else ml_result["ml_direction"]

    confidence_score = round(final_conf * 10, 1)
    if confidence_score < 1.0:
        direction = "NEUTRAL"
        confidence_score = max(3.0, round(final_conf * 5, 1))

    label = "HIGH" if confidence_score >= 7.0 else "MEDIUM" if confidence_score >= 5.0 else "LOW"

    atr = signals.get("atr_value") or 15.0
    stop_dir = direction if direction != "NEUTRAL" else ("BULLISH" if ta_bull > ta_bear else "BEARISH")
    stops = calculate_gold_stop_loss(atr, stop_dir, current_price)

    return {
        "direction": direction, "confidence": confidence_score,
        "confidence_label": label, "entry_price": current_price,
        "stop_loss": stops["stop_loss"], "take_profit_1": stops["take_profit_1"],
        "take_profit_2": stops["take_profit_2"], "take_profit_3": stops["take_profit_3"],
        "stop_distance": stops["stop_distance"], "timeframe": "30min",
        "ta_bull_score": round(ta_bull, 1), "ta_bear_score": round(ta_bear, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── 5. Format Output ─────────────────────────────────────────────────────

def format_output(final, ml_result, signals):
    now = datetime.now(timezone.utc)
    arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "◆"}[final["direction"]]
    e = final["entry_price"]

    L = []
    L.append("=" * 55)
    L.append(f"  XAUUSD 30-MIN FORECAST  |  {now.strftime('%Y-%m-%d %H:%M')} UTC")
    L.append("=" * 55)
    L.append(f"")
    L.append(f"  {arrow} {final['direction']}  |  {final['confidence']}/10 ({final['confidence_label']})")
    L.append(f"  Entry: ${e:,.2f}")
    L.append(f"  SL:    ${final['stop_loss']:,.2f}  ({final['stop_distance']:.1f} pts)")
    L.append(f"  TP1:   ${final['take_profit_1']:,.2f}  |  TP2: ${final['take_profit_2']:,.2f}")
    L.append(f"  TP3:   ${final['take_profit_3']:,.2f}")

    L.append(f"")
    L.append(f"  ── ML Ensemble (40%) ──")
    if ml_result:
        L.append(f"  XGBoost:      {ml_result['xgb_prob']:.3f}")
        L.append(f"  LightGBM:     {ml_result['lgbm_prob']:.3f}")
        L.append(f"  CatBoost:     {ml_result['cb_prob']:.3f}")
        L.append(f"  Meta-Learner: {ml_result['ml_prob']:.3f} → {ml_result['ml_direction']}")
        L.append(f"  Features:     {ml_result['features_available']}/{ml_result['features_total']}")
    else:
        L.append(f"  ⚠ ML unavailable (TA only)")

    L.append(f"")
    L.append(f"  ── Technical Analysis (60%) ──")
    L.append(f"  Price:    ${signals.get('price', 0):,.2f}")
    L.append(f"  EMA 9/21: {signals.get('ema_cross', 'N/A')}")
    L.append(f"  RSI(14):  {signals.get('rsi_value', 0):.1f} — {signals.get('rsi_signal', 'N/A')}")
    L.append(f"  MACD:     {signals.get('macd_signal', 'N/A')}")
    L.append(f"  MACD Div: {signals.get('macd_divergence', 'N/A')}")
    L.append(f"  RSI Div:  {signals.get('rsi_divergence', 'N/A')}")
    L.append(f"  BB:       {signals.get('bb_signal', 'N/A')} ({signals.get('bb_position', 0):.2f})")
    L.append(f"  Stoch:    {signals.get('stoch_signal', 'N/A')}")
    L.append(f"  Trend:    {signals.get('trend', 'N/A')}")
    L.append(f"  ATR(14):  {signals.get('atr_value', 0):.2f}")
    L.append(f"  Bull/BEAR: {final['ta_bull_score']:.1f} / {final['ta_bear_score']:.1f}")

    if "session" in signals:
        s = signals["session"]
        L.append(f"")
        L.append(f"  ── Session ──")
        L.append(f"  {s['current_session']} (Vol: {s['session_volatility']})")
        if s.get("gold_event"):
            L.append(f"  Event: {s['gold_event']}")

    if signals.get("sma_20"):
        L.append(f"")
        L.append(f"  ── Key Levels ──")
        L.append(f"  SMA 20:  ${signals['sma_20']:,.2f}")
        if signals.get("sma_50"):
            L.append(f"  SMA 50:  ${signals['sma_50']:,.2f}")
        if signals.get("sma_200"):
            L.append(f"  SMA 200: ${signals['sma_200']:,.2f}")

    L.append("")
    L.append("=" * 55)
    return "\n".join(L)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("\nFetching real-time XAUUSD data...")

    try:
        df_m15 = fetch_candles("15min", 500)
        print(f"  M15: {len(df_m15)} candles, last={df_m15.index[-1]}")
    except Exception as e:
        print(f"  ERROR fetching M15: {e}")
        return

    try:
        df_h1 = fetch_candles("1h", 1000)
        print(f"  H1:  {len(df_h1)} candles, last={df_h1.index[-1]}")
    except Exception as e:
        print(f"  WARNING: H1 unavailable: {e}")
        df_h1 = None

    try:
        current_price = fetch_current_price()
        print(f"  Live Price: ${current_price:,.2f}")
    except:
        current_price = float(df_m15.iloc[-1]["close"])

    print("\nRunning technical analysis...")
    signals, _ = compute_technical_analysis(df_m15)

    print("Running ML ensemble prediction...")
    ml_result = None
    try:
        ml_result = compute_ml_prediction(df_m15, df_h1)
        print(f"  ML: {ml_result['ml_direction']} (prob={ml_result['ml_prob']:.3f})")
    except Exception as e:
        print(f"  ML Error: {e}")

    final = combine_signals(ml_result, signals, current_price)
    report = format_output(final, ml_result, signals)
    print(report)

    # Save prediction
    memory_dir = os.path.join(BASE_DIR, "memory")
    os.makedirs(memory_dir, exist_ok=True)
    pred_path = os.path.join(memory_dir, "latest_prediction_30m.json")
    safe_signals = {k: v for k, v in signals.items() if not isinstance(v, (dict, type(pd.Series())))}
    with open(pred_path, "w") as f:
        json.dump({"final": final, "ml": ml_result, "signals": safe_signals,
                    "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2, default=str)
    print(f"Prediction saved → {pred_path}")

    return final


if __name__ == "__main__":
    main()
