"""
XAUUSD 5-Min Prediction Pipeline
Load trained models and predict direction from live data.
"""
import pandas as pd
import numpy as np
import ta
import pickle
import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_models():
    with open(f'{MODEL_DIR}/xgb_model.pkl', 'rb') as f: xgb = pickle.load(f)
    with open(f'{MODEL_DIR}/rf_model.pkl', 'rb') as f: rf = pickle.load(f)
    with open(f'{MODEL_DIR}/lr_model.pkl', 'rb') as f: lr = pickle.load(f)
    with open(f'{MODEL_DIR}/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open(f'{MODEL_DIR}/feature_cols.json', 'r') as f: feature_cols = json.load(f)
    return xgb, rf, lr, scaler, feature_cols


def engineer_features(df):
    """Engineer all 89 technical indicator features from OHLCV data."""
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']

    features = pd.DataFrame(index=df.index)

    for w in [5, 10, 20, 50, 100, 200]:
        features[f'sma_{w}'] = ta.trend.sma_indicator(close, window=w)
        features[f'ema_{w}'] = ta.trend.ema_indicator(close, window=w)

    features['ema_5_20_cross'] = features['ema_5'] - features['ema_20']
    features['ema_10_50_cross'] = features['ema_10'] - features['ema_50']
    for w in [10, 20, 50, 200]:
        features[f'close_vs_sma{w}_pct'] = (close - features[f'sma_{w}']) / features[f'sma_{w}'] * 100

    for w in [7, 14, 21]:
        features[f'rsi_{w}'] = ta.momentum.rsi(close, window=w)

    macd = ta.trend.MACD(close)
    features['macd_line'] = macd.macd()
    features['macd_signal'] = macd.macd_signal()
    features['macd_hist'] = macd.macd_diff()

    for w in [20, 50]:
        bb = ta.volatility.BollingerBands(close, window=w, window_dev=2)
        features[f'bb_upper_{w}'] = bb.bollinger_hband()
        features[f'bb_lower_{w}'] = bb.bollinger_lband()
        features[f'bb_width_{w}'] = (features[f'bb_upper_{w}'] - features[f'bb_lower_{w}']) / features[f'bb_upper_{w}'] * 100
        features[f'bb_pct_{w}'] = (close - features[f'bb_lower_{w}']) / (features[f'bb_upper_{w}'] - features[f'bb_lower_{w}']) * 100

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    features['stoch_k'] = stoch.stoch()
    features['stoch_d'] = stoch.stoch_signal()
    features['stoch_k_d_diff'] = features['stoch_k'] - features['stoch_d']

    for w in [7, 14, 21]:
        features[f'atr_{w}'] = ta.volatility.average_true_range(high, low, close, window=w)
    features['atr_14_pct'] = features['atr_14'] / close * 100

    adx = ta.trend.ADXIndicator(high, low, close)
    features['adx'] = adx.adx()
    features['di_pos'] = adx.adx_pos()
    features['di_neg'] = adx.adx_neg()
    features['di_diff'] = features['di_pos'] - features['di_neg']

    for w in [14, 20]:
        features[f'cci_{w}'] = ta.trend.cci(high, low, close, window=w)

    features['williams_r'] = ta.momentum.williams_r(high, low, close)

    for w in [3, 5, 10, 20]:
        features[f'roc_{w}'] = ta.momentum.roc(close, window=w)

    for w in [3, 5, 10]:
        features[f'momentum_{w}'] = close - close.shift(w)

    features['candle_body'] = close - open_
    features['candle_range'] = high - low
    features['upper_shadow'] = high - np.maximum(close, open_)
    features['lower_shadow'] = np.minimum(close, open_) - low
    features['body_to_range'] = features['candle_body'] / features['candle_range'].replace(0, np.nan)

    features['vol_sma_10'] = volume.rolling(10).mean()
    features['vol_sma_50'] = volume.rolling(50).mean()
    features['vol_ratio'] = volume / features['vol_sma_10'].replace(0, np.nan)
    features['obv'] = ta.volume.on_balance_volume(close, volume)

    for lag in [1, 2, 3, 5]:
        features[f'close_lag_{lag}'] = close.shift(lag)
        features[f'return_lag_{lag}'] = close.pct_change(lag)

    for w in [10, 20, 50]:
        features[f'close_std_{w}'] = close.rolling(w).std()
        features[f'close_skew_{w}'] = close.rolling(w).skew()

    features['hl_ratio'] = (high / low.replace(0, np.nan) - 1) * 100

    if 'datetime' in df.columns:
        features['hour'] = df['datetime'].dt.hour
        features['day_of_week'] = df['datetime'].dt.dayofweek
        features['month'] = df['datetime'].dt.month
        features['session_asian'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['session_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['session_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
        features['session_overlap'] = ((features['hour'] >= 13) & (features['hour'] < 16)).astype(int)

    return features


def predict(df, model_name='xgb'):
    """
    Predict next-candle direction from OHLCV DataFrame.

    Parameters:
        df: DataFrame with [datetime, open, high, low, close, volume]
            Must have at least 200 rows of history.
        model_name: 'xgb', 'rf', 'lr'

    Returns:
        dict with direction, confidence, probabilities
    """
    xgb, rf, lr, scaler, feature_cols = load_models()

    features = engineer_features(df)
    features = features.dropna()

    if len(features) == 0:
        return {"error": "Not enough data. Need at least 200 candles of history."}

    latest = features.iloc[[-1]][feature_cols].values

    if model_name == 'lr':
        latest_scaled = scaler.transform(latest)
        proba = lr.predict_proba(latest_scaled)[0]
        pred = lr.predict(latest_scaled)[0]
    elif model_name == 'rf':
        proba = rf.predict_proba(latest)[0]
        pred = rf.predict(latest)[0]
    else:
        proba = xgb.predict_proba(latest)[0]
        pred = xgb.predict(latest)[0]

    direction = 'UP' if pred == 1 else 'DOWN'
    confidence = float(np.max(proba))

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "up_probability": round(float(proba[1]), 4),
        "down_probability": round(float(proba[0]), 4),
        "current_price": float(df['close'].iloc[-1]),
        "model": model_name,
        "timestamp": str(df['datetime'].iloc[-1]) if 'datetime' in df.columns else None
    }


if __name__ == '__main__':
    import yfinance as yf
    data = yf.download('GC=F', period='60d', interval='5m', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    data = data.reset_index()
    data.columns = ['datetime', 'close', 'high', 'low', 'open', 'volume']
    data = data[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    for model_name in ['xgb', 'rf', 'lr']:
        result = predict(data, model_name=model_name)
        print(f"{model_name}: {result['direction']} (conf: {result['confidence']:.3f}, up: {result['up_probability']:.3f})")
