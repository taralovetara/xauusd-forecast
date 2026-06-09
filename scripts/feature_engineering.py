"""
Feature Engineering for XAUUSD 5-min data.
Generates 89 technical indicator features from OHLCV data.
"""
import pandas as pd
import numpy as np
import ta
import os

INPUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'xauusd_5m_merged.csv')
OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'xauusd_features.pkl')

def engineer_features(df):
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']

    # Moving Averages
    for w in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{w}'] = ta.trend.sma_indicator(close, window=w)
        df[f'ema_{w}'] = ta.trend.ema_indicator(close, window=w)

    # Crosses
    df['ema_5_20_cross'] = (df['ema_5'] - df['ema_20']).astype(float)
    df['ema_10_50_cross'] = (df['ema_10'] - df['ema_50']).astype(float)
    for w in [10, 20, 50, 200]:
        df[f'close_vs_sma{w}_pct'] = ((close - df[f'sma_{w}']) / df[f'sma_{w}'] * 100).astype(float)

    # RSI
    for w in [7, 14, 21]:
        df[f'rsi_{w}'] = ta.momentum.rsi(close, window=w)

    # MACD
    macd = ta.trend.MACD(close)
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # Bollinger Bands
    for w in [20, 50]:
        bb = ta.volatility.BollingerBands(close, window=w, window_dev=2)
        df[f'bb_upper_{w}'] = bb.bollinger_hband()
        df[f'bb_lower_{w}'] = bb.bollinger_lband()
        df[f'bb_width_{w}'] = ((df[f'bb_upper_{w}'] - df[f'bb_lower_{w}']) / df[f'bb_upper_{w}'] * 100).astype(float)
        df[f'bb_pct_{w}'] = ((close - df[f'bb_lower_{w}']) / (df[f'bb_upper_{w}'] - df[f'bb_lower_{w}']) * 100).astype(float)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_k_d_diff'] = (df['stoch_k'] - df['stoch_d']).astype(float)

    # ATR
    for w in [7, 14, 21]:
        df[f'atr_{w}'] = ta.volatility.average_true_range(high, low, close, window=w)
    df['atr_14_pct'] = (df['atr_14'] / close * 100).astype(float)

    # ADX
    adx = ta.trend.ADXIndicator(high, low, close)
    df['adx'] = adx.adx()
    df['di_pos'] = adx.adx_pos()
    df['di_neg'] = adx.adx_neg()
    df['di_diff'] = (df['di_pos'] - df['di_neg']).astype(float)

    # CCI
    for w in [14, 20]:
        df[f'cci_{w}'] = ta.trend.cci(high, low, close, window=w)

    # Williams %R
    df['williams_r'] = ta.momentum.williams_r(high, low, close)

    # ROC
    for w in [3, 5, 10, 20]:
        df[f'roc_{w}'] = ta.momentum.roc(close, window=w)

    # Momentum
    for w in [3, 5, 10]:
        df[f'momentum_{w}'] = (close - close.shift(w)).astype(float)

    # Candle features
    df['candle_body'] = (close - open_).astype(float)
    df['candle_range'] = (high - low).astype(float)
    df['upper_shadow'] = (high - np.maximum(close, open_)).astype(float)
    df['lower_shadow'] = (np.minimum(close, open_) - low).astype(float)
    df['body_to_range'] = (df['candle_body'] / df['candle_range'].replace(0, np.nan)).astype(float)

    # Volume
    df['vol_sma_10'] = volume.rolling(10).mean()
    df['vol_sma_50'] = volume.rolling(50).mean()
    df['vol_ratio'] = (volume / df['vol_sma_10'].replace(0, np.nan)).astype(float)
    df['obv'] = ta.volume.on_balance_volume(close, volume)

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = close.shift(lag).astype(float)
        df[f'return_lag_{lag}'] = close.pct_change(lag).astype(float)

    # Rolling stats
    for w in [10, 20, 50]:
        df[f'close_std_{w}'] = close.rolling(w).std()
        df[f'close_skew_{w}'] = close.rolling(w).skew()

    df['hl_ratio'] = ((high / low.replace(0, np.nan)) - 1).astype(float) * 100

    # Session
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['session_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['session_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['session_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['session_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)

    # Target
    df['next_close'] = df['close'].shift(-1)
    df['target_direction'] = (df['next_close'] > df['close']).astype(int)
    df['target_return'] = (df['next_close'] - df['close']) / df['close'] * 100

    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = pd.read_csv(INPUT, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = engineer_features(df)
    df.to_pickle(OUTPUT)
    print(f"Features: {len(df):,} rows, {len(df.columns)} columns → {OUTPUT}")
