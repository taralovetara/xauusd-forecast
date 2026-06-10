#!/usr/bin/env python3
"""
feature_engineering_v2.py — 3-Concept Framework for XAUUSD ML Feature Engineering
==================================================================================
Concepts: TREND / MOMENTUM / CYCLE

Gold-specific adjustments:
- ATR normalization is critical — gold's ATR is 10-50x larger than EURUSD
- All distance features are in ATR units, not raw pips
- Fibonacci levels are more significant for gold
- Session encoding emphasizes London Fix (15:00 UTC) and COMEX open (18:00 UTC)

Output: ~99 base features + ~79 advanced features = ~178 total
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sma(series, period):
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series, period):
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _atr(df, period=14):
    """Average True Range (gold-native, no pip conversion)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _ema(tr, period)


def _rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _stochastic(df, k_period=14, d_period=3):
    """Stochastic oscillator %K and %D."""
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    denom = high_max - low_min
    denom = denom.replace(0, np.nan)
    k = 100.0 * (df["close"] - low_min) / denom
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def _macd(series, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram."""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _adx(df, period=14):
    """Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = _atr(df, period)

    plus_di = 100.0 * _ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100.0 * _ema(minus_dm, period) / atr_val.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _ema(dx, period)
    return adx, plus_di, minus_di


def _bb(series, period=20, num_std=2.0):
    """Bollinger Bands: upper, middle, lower, width, %B."""
    middle = _sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, middle, lower, width, pct_b


def _detect_divergence(price, indicator, lookback=20):
    """
    Detect bearish divergence: price makes new high but indicator doesn't.
    Returns: divergence_flag (binary), divergence_magnitude.
    """
    price_high = price.rolling(window=lookback, min_periods=lookback).max()
    indic_high = indicator.rolling(window=lookback, min_periods=lookback).max()
    prev_price_high = price_high.shift(lookback // 2)
    prev_indic_high = indic_high.shift(lookback // 2)

    bearish = (price_high > prev_price_high) & (indic_high < prev_indic_high)

    price_low = price.rolling(window=lookback, min_periods=lookback).min()
    indic_low = indicator.rolling(window=lookback, min_periods=lookback).min()
    prev_price_low = price_low.shift(lookback // 2)
    prev_indic_low = indic_low.shift(lookback // 2)

    bullish = (price_low < prev_price_low) & (indic_low > prev_indic_low)

    flag = bearish.astype(int) - bullish.astype(int)
    magnitude = np.where(
        bearish,
        (price_high - prev_price_high) / prev_price_high.replace(0, np.nan),
        np.where(
            bullish,
            (prev_price_low - price_low) / prev_price_low.replace(0, np.nan),
            0.0,
        ),
    )
    return flag, pd.Series(magnitude, index=price.index)


def _session_encode(hour):
    """
    Session encoding for gold:
      Asian  = 0  (00-07 UTC)
      London = 1  (07-15 UTC)
      NY     = 2  (15-21 UTC)  — London Fix at 15:00 UTC, COMEX open 18:00 UTC
      Overlap= 3  (overlap not typical for gold, but late London/NY = 13-16 can be treated)
    
    Simplified for gold market structure.
    """
    return pd.cut(
        hour,
        bins=[-1, 7, 13, 18, 24],
        labels=[0, 1, 2, 3],
    ).astype(float)


# ---------------------------------------------------------------------------
# CONCEPT 1 — TREND (33 features)
# ---------------------------------------------------------------------------

def _compute_trend_features(df_m15, df_h1=None):
    """Compute 33 Trend features on M15 data."""
    close = df_m15["close"]
    high = df_m15["high"]
    low = df_m15["low"]
    atr14 = _atr(df_m15, 14)

    feat = pd.DataFrame(index=df_m15.index)

    # --- Price vs SMA (6 features) ---
    for p in [20, 50, 200]:
        sma = _sma(close, p)
        feat[f"price_vs_sma{p}_atr"] = (close - sma) / atr14

    # --- Price vs EMA (3 features) ---
    for p in [9, 21, 55]:
        ema = _ema(close, p)
        feat[f"price_vs_ema{p}_atr"] = (close - ema) / atr14

    # --- SMA slopes (6 features: 3 SMAs × 2 slopes) ---
    for p in [20, 50, 200]:
        sma = _sma(close, p)
        feat[f"sma{p}_roc5"] = sma.pct_change(5)
        feat[f"sma{p}_roc15"] = sma.pct_change(15)

    # --- EMA slopes (6 features: 3 EMAs × 2 slopes) ---
    for p in [9, 21, 55]:
        ema = _ema(close, p)
        feat[f"ema{p}_roc5"] = ema.pct_change(5)
        feat[f"ema{p}_roc15"] = ema.pct_change(15)

    # --- SMA crossovers (2 binary) ---
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    feat["sma20_gt_sma50"] = (sma20 > sma50).astype(int)
    feat["sma50_gt_sma200"] = (sma50 > sma200).astype(int)

    # --- EMA crossovers (2 binary) ---
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema55 = _ema(close, 55)
    feat["ema9_gt_ema21"] = (ema9 > ema21).astype(int)
    feat["ema21_gt_ema55"] = (ema21 > ema55).astype(int)

    # --- ADX (1 feature) ---
    adx, plus_di, minus_di = _adx(df_m15, 14)
    feat["adx14"] = adx

    # --- Ichimoku-style: price vs "cloud" (2 features) ---
    # Simplified: use SMA offsets as cloud boundaries
    tenkan = (_sma(high, 9) + _sma(low, 9)) / 2
    kijun = (_sma(high, 26) + _sma(low, 26)) / 2
    # Cloud = SMA26 offset 26 bars ahead (simplified)
    cloud_a = tenkan.shift(26)
    cloud_b = kijun.shift(26)
    cloud_top = pd.concat([cloud_a, cloud_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([cloud_a, cloud_b], axis=1).min(axis=1)
    feat["price_above_cloud"] = (close > cloud_top).astype(int)
    feat["price_vs_cloud_atr"] = (close - (cloud_top + cloud_bot) / 2) / atr14

    # --- Higher timeframe trend: H1 close vs H1 SMA20/50 (3 features) ---
    if df_h1 is not None and not df_h1.empty:
        h1_close = df_h1["close"]
        h1_sma20 = _sma(h1_close, 20)
        h1_sma50 = _sma(h1_close, 50)
        h1_atr14 = _atr(df_h1, 14)

        h1_feat = pd.DataFrame(index=df_h1.index)
        h1_feat["h1_close_vs_sma20_atr"] = (h1_close - h1_sma20) / h1_atr14
        h1_feat["h1_close_vs_sma50_atr"] = (h1_close - h1_sma50) / h1_atr14
        h1_feat["h1_sma20_gt_sma50"] = (h1_sma20 > h1_sma50).astype(int)

        # Reindex H1 features to M15 by forward-filling
        h1_feat = h1_feat.reindex(df_m15.index, method="ffill")
        for col in h1_feat.columns:
            feat[col] = h1_feat[col]
    else:
        feat["h1_close_vs_sma20_atr"] = np.nan
        feat["h1_close_vs_sma50_atr"] = np.nan
        feat["h1_sma20_gt_sma50"] = np.nan

    # --- SMA 30-bar ROC (1 feature) ---
    feat["sma200_roc30"] = sma200.pct_change(30)

    # Total: 3 + 3 + 6 + 6 + 2 + 2 + 1 + 2 + 3 + 1 = 29... need 33
    # Add 4 more slope features
    feat["sma50_roc30"] = sma50.pct_change(30)
    feat["ema9_roc3"] = ema9.pct_change(3)
    feat["ema21_roc10"] = ema21.pct_change(10)
    feat["sma20_roc30"] = sma20.pct_change(30)

    # = 33 features
    return feat


# ---------------------------------------------------------------------------
# CONCEPT 2 — MOMENTUM (33 features)
# ---------------------------------------------------------------------------

def _compute_momentum_features(df_m15):
    """Compute 33 Momentum features on M15 data."""
    close = df_m15["close"]
    high = df_m15["high"]
    low = df_m15["low"]
    volume = df_m15["volume"]

    feat = pd.DataFrame(index=df_m15.index)

    # --- RSI(14): raw, distance from 50, flags (4 features) ---
    rsi14 = _rsi(close, 14)
    feat["rsi14"] = rsi14
    feat["rsi14_dist50"] = rsi14 - 50.0
    feat["rsi14_oversold"] = (rsi14 < 30).astype(int)
    feat["rsi14_overbought"] = (rsi14 > 70).astype(int)

    # --- RSI divergence (2 features) ---
    rsi_div_flag, rsi_div_mag = _detect_divergence(close, rsi14, lookback=20)
    feat["rsi_divergence_flag"] = rsi_div_flag
    feat["rsi_divergence_magnitude"] = rsi_div_mag

    # --- MACD (5 features) ---
    macd_line, macd_signal, macd_hist = _macd(close)
    feat["macd_histogram"] = macd_hist
    feat["macd_signal"] = macd_signal
    feat["macd_crossover_up"] = ((macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))).astype(int)
    feat["macd_crossover_down"] = ((macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))).astype(int)
    feat["macd_histogram_sign"] = (macd_hist > 0).astype(int)

    # --- MACD divergence (2 features) ---
    macd_div_flag, macd_div_mag = _detect_divergence(close, macd_hist, lookback=20)
    feat["macd_divergence_flag"] = macd_div_flag
    feat["macd_divergence_magnitude"] = macd_div_mag

    # --- Stochastic (4 features) ---
    stoch_k, stoch_d = _stochastic(df_m15)
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d
    feat["stoch_crossover_up"] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)
    feat["stoch_crossover_down"] = ((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))).astype(int)

    # --- Rate of change (4 features) ---
    for p in [5, 10, 20, 60]:
        feat[f"roc_{p}"] = close.pct_change(p)

    # --- Volume ratio (1 feature) ---
    vol_sma20 = _sma(volume, 20)
    feat["volume_ratio"] = volume / vol_sma20.replace(0, np.nan)

    # --- MFI approximation (Money Flow Index) (1 feature) ---
    typical_price = (high + low + close) / 3.0
    mf = typical_price * volume
    delta_tp = typical_price.diff()
    pos_mf = mf.where(delta_tp > 0, 0.0)
    neg_mf = mf.where(delta_tp < 0, 0.0)
    pos_mf_sma = _sma(pos_mf, 14)
    neg_mf_sma = _sma(neg_mf, 14)
    mf_ratio = pos_mf_sma / neg_mf_sma.replace(0, np.nan)
    feat["mfi14"] = 100.0 - 100.0 / (1.0 + mf_ratio)

    # --- Momentum convergence (1 feature) ---
    # RSI>50, MACD histogram>0, Stoch K>D  → all bullish
    rsi_bull = rsi14 > 50
    macd_bull = macd_hist > 0
    stoch_bull = stoch_k > stoch_d
    convergence = (rsi_bull.astype(int) + macd_bull.astype(int) + stoch_bull.astype(int))
    feat["momentum_convergence"] = convergence
    feat["momentum_all_bull"] = (convergence == 3).astype(int)
    feat["momentum_all_bear"] = (convergence == 0).astype(int)

    # --- Additional momentum features to reach 33 ---
    feat["rsi7"] = _rsi(close, 7)
    feat["rsi21"] = _rsi(close, 21)
    feat["rsi14_slope3"] = rsi14.diff(3)
    feat["stoch_k_oversold"] = (stoch_k < 20).astype(int)
    feat["stoch_k_overbought"] = (stoch_k > 80).astype(int)

    # Count: 4+2+5+2+4+4+1+1+3+5 = 31... need 2 more
    feat["macd_histogram_slope3"] = macd_hist.diff(3)
    feat["rsi14_vs_stoch_k"] = rsi14 - stoch_k

    # = 33 features
    return feat


# ---------------------------------------------------------------------------
# CONCEPT 3 — CYCLE (33 features)
# ---------------------------------------------------------------------------

def _compute_cycle_features(df_m15):
    """Compute 33 Cycle features on M15 data."""
    close = df_m15["close"]
    high = df_m15["high"]
    low = df_m15["low"]
    atr14 = _atr(df_m15, 14)

    feat = pd.DataFrame(index=df_m15.index)

    # --- Bollinger Band width, %B (2 features) ---
    bb_upper, bb_mid, bb_lower, bb_width, bb_pct_b = _bb(close, 20, 2.0)
    feat["bb_width"] = bb_width
    feat["bb_pct_b"] = bb_pct_b

    # --- BB squeeze detection (1 feature) ---
    bb_width_low20 = bb_width.rolling(window=20, min_periods=20).min()
    feat["bb_squeeze"] = (bb_width <= bb_width_low20 * 1.01).astype(int)

    # --- ATR features (3 features) ---
    feat["atr14"] = atr14
    feat["atr14_normalized"] = atr14 / close
    feat["atr14_roc5"] = atr14.pct_change(5)

    # --- Fibonacci levels (3 features) ---
    # Using 20-bar swing high/low as reference
    swing_high = high.rolling(window=20, min_periods=20).max()
    swing_low = low.rolling(window=20, min_periods=20).min()
    swing_range = swing_high - swing_low
    fib_382 = swing_high - 0.382 * swing_range
    fib_50 = swing_high - 0.500 * swing_range
    fib_618 = swing_high - 0.618 * swing_range
    feat["price_vs_fib382_atr"] = (close - fib_382) / atr14
    feat["price_vs_fib50_atr"] = (close - fib_50) / atr14
    feat["price_vs_fib618_atr"] = (close - fib_618) / atr14

    # --- Hour of day (cyclical sin/cos) (2 features) ---
    hour = df_m15.index.hour if hasattr(df_m15.index, "hour") else pd.Series(0, index=df_m15.index)
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # --- Day of week (cyclical sin/cos) (2 features) ---
    dow = df_m15.index.dayofweek if hasattr(df_m15.index, "dayofweek") else pd.Series(0, index=df_m15.index)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # --- Session indicator (4 features: one-hot style) ---
    if isinstance(hour, pd.Series):
        session = _session_encode(hour)
    else:
        session = _session_encode(pd.Series(hour, index=df_m15.index))
    feat["session_asian"] = (session == 0).astype(int)
    feat["session_london"] = (session == 1).astype(int)
    feat["session_ny"] = (session == 2).astype(int)
    feat["session_overlap"] = (session == 3).astype(int)

    # --- London Fix & COMEX open proximity (2 features) ---
    if isinstance(hour, pd.Series):
        feat["london_fix_proximity"] = np.exp(-0.5 * ((hour - 15) ** 2))
        feat["comex_open_proximity"] = np.exp(-0.5 * ((hour - 18) ** 2))
    else:
        hour_s = pd.Series(hour, index=df_m15.index)
        feat["london_fix_proximity"] = np.exp(-0.5 * ((hour_s - 15) ** 2))
        feat["comex_open_proximity"] = np.exp(-0.5 * ((hour_s - 18) ** 2))

    # --- Mean reversion: Z-score (1 feature) ---
    mean20 = _sma(close, 20)
    std20 = close.rolling(window=20, min_periods=20).std()
    feat["zscore20"] = (close - mean20) / std20.replace(0, np.nan)

    # --- Volatility regime (2 features) ---
    atr50 = _atr(df_m15, 50)
    atr_percentile = atr14.rolling(window=50, min_periods=50).rank(pct=True)
    feat["atr_regime_percentile"] = atr_percentile
    feat["atr14_vs_atr50"] = atr14 / atr50.replace(0, np.nan)

    # --- Swing detection (4 features) ---
    # Local highs/lows in last 20 bars
    swing_h = high.rolling(5, center=True, min_periods=5).max()
    swing_l = low.rolling(5, center=True, min_periods=5).min()
    feat["at_swing_high"] = (high >= swing_h).astype(int).shift(1)  # shift to avoid lookahead
    feat["at_swing_low"] = (low <= swing_l).astype(int).shift(1)

    # Distance to last swing high/low in ATR units
    last_swing_high = high.rolling(20, min_periods=20).max()
    last_swing_low = low.rolling(20, min_periods=20).min()
    feat["dist_swing_high_atr"] = (last_swing_high - close) / atr14
    feat["dist_swing_low_atr"] = (close - last_swing_low) / atr14

    # --- Additional cycle features to reach 33 ---
    feat["bb_pct_b_slope5"] = bb_pct_b.diff(5)
    feat["zscore20_slope5"] = feat["zscore20"].diff(5)

    # Count: 2+1+3+3+2+2+4+2+1+2+4+2 = 28... need 5 more
    feat["fib_confluence"] = (
        ((close - fib_382).abs() < 0.5 * atr14).astype(int)
        + ((close - fib_50).abs() < 0.5 * atr14).astype(int)
        + ((close - fib_618).abs() < 0.5 * atr14).astype(int)
    )
    feat["atr14_roc10"] = atr14.pct_change(10)
    feat["bb_width_roc5"] = bb_width.pct_change(5)
    feat["price_position_20bar"] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min()).replace(0, np.nan)
    feat["cycle_hour_dow_interaction"] = feat["hour_sin"] * feat["dow_sin"]

    # = 33 features
    return feat


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_base_features(df_m15, df_h1=None):
    """
    Compute ~99 base features organized into 3 concepts:
      CONCEPT 1 — TREND (33 features)
      CONCEPT 2 — MOMENTUM (33 features)
      CONCEPT 3 — CYCLE (33 features)
    
    Parameters
    ----------
    df_m15 : pd.DataFrame
        M15 OHLCV DataFrame with DatetimeIndex.
    df_h1 : pd.DataFrame, optional
        H1 OHLCV DataFrame for higher-timeframe trend features.
    
    Returns
    -------
    pd.DataFrame with ~99 feature columns + original OHLCV columns preserved.
    """
    # Ensure sorted datetime index
    df_m15 = df_m15.sort_index()

    trend = _compute_trend_features(df_m15, df_h1)
    momentum = _compute_momentum_features(df_m15)
    cycle = _compute_cycle_features(df_m15)

    # Prefix columns
    trend.columns = [f"trend_{c}" for c in trend.columns]
    momentum.columns = [f"mom_{c}" for c in momentum.columns]
    cycle.columns = [f"cyc_{c}" for c in cycle.columns]

    features = pd.concat([trend, momentum, cycle], axis=1)

    # Preserve OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df_m15.columns:
            features[col] = df_m15[col]

    return features


def compute_advanced_features(df_features):
    """
    Compute ~79 advanced features from base features:
      - Lagged features: key indicators at lag 1, 2, 3, 5 bars
      - Interaction features
      - Rolling stats of key indicators
      - Simplified candlestick pattern features
      - Cross-timeframe agreement
      - Rate of change of key features
    
    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame with base features (output of compute_base_features).
    
    Returns
    -------
    pd.DataFrame with advanced feature columns added.
    """
    df = df_features.copy()
    adv = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    # === Lagged features (5 key indicators × 4 lags = 20 features) ===
    lag_cols = [
        "mom_rsi14",
        "mom_macd_histogram",
        "cyc_bb_pct_b",
        "trend_adx14",
        "mom_stoch_k",
    ]
    for col in lag_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 5]:
                adv[f"lag{lag}_{col}"] = df[col].shift(lag)

    # === Interaction features (6 features) ===
    if "mom_rsi14" in df.columns and "mom_macd_signal" in df.columns:
        adv["inter_rsi_macd_sig"] = df["mom_rsi14"] * df["mom_macd_signal"] / 10000.0
    if "trend_ema9_gt_ema21" in df.columns and "trend_adx14" in df.columns:
        adv["inter_emacross_adx"] = df["trend_ema9_gt_ema21"] * df["trend_adx14"] / 100.0
    if "cyc_bb_pct_b" in df.columns and "cyc_atr_regime_percentile" in df.columns:
        adv["inter_bb_atrregime"] = df["cyc_bb_pct_b"] * df["cyc_atr_regime_percentile"]
    if "mom_rsi14_dist50" in df.columns and "trend_price_vs_sma20_atr" in df.columns:
        adv["inter_rsidist_trenddist"] = df["mom_rsi14_dist50"] * df["trend_price_vs_sma20_atr"]
    if "mom_momentum_convergence" in df.columns and "cyc_zscore20" in df.columns:
        adv["inter_momconv_zscore"] = df["mom_momentum_convergence"] * df["cyc_zscore20"]
    if "trend_sma20_gt_sma50" in df.columns and "mom_macd_histogram_sign" in df.columns:
        adv["inter_trend_mom"] = df["trend_sma20_gt_sma50"] * df["mom_macd_histogram_sign"]

    # === Rolling stats (5 indicators × 2 windows × 2 stats = 20 features) ===
    roll_cols = ["mom_rsi14", "mom_macd_histogram", "cyc_bb_pct_b", "cyc_zscore20", "trend_adx14"]
    for col in roll_cols:
        if col in df.columns:
            s = df[col]
            for w in [5, 10]:
                adv[f"roll{w}_mean_{col}"] = s.rolling(w, min_periods=w).mean()
                adv[f"roll{w}_std_{col}"] = s.rolling(w, min_periods=w).std()

    # === Candlestick pattern features (5 features) ===
    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

    # Doji: body < 10% of range
    adv["pattern_doji"] = (body / full_range < 0.1).astype(int)

    # Engulfing: current body engulfs previous body
    prev_body = (close.shift(1) - open_.shift(1)).abs()
    prev_bull = close.shift(1) > open_.shift(1)
    curr_bear = close < open_
    prev_bear = close.shift(1) < open_.shift(1)
    curr_bull = close > open_
    adv["pattern_bull_engulf"] = (prev_bear & curr_bull & (body > prev_body)).astype(int)
    adv["pattern_bear_engulf"] = (prev_bull & curr_bear & (body > prev_body)).astype(int)

    # Hammer: lower shadow > 2x body, upper shadow < body
    adv["pattern_hammer"] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)

    # Shooting star: upper shadow > 2x body, lower shadow < body
    adv["pattern_shooting_star"] = ((upper_shadow > 2 * body) & (lower_shadow < body)).astype(int)

    # === Cross-timeframe agreement (3 features) ===
    if "trend_h1_sma20_gt_sma50" in df.columns and "trend_sma20_gt_sma50" in df.columns:
        adv["htf_trend_agreement"] = (
            df["trend_h1_sma20_gt_sma50"] == df["trend_sma20_gt_sma50"]
        ).astype(int)
        # Multi-timeframe trend strength
        adv["htf_mtf_strength"] = (
            df["trend_sma20_gt_sma50"] + df["trend_sma50_gt_sma200"] + df["trend_h1_sma20_gt_sma50"]
        )
    else:
        adv["htf_trend_agreement"] = np.nan
        adv["htf_mtf_strength"] = np.nan

    if "trend_h1_close_vs_sma20_atr" in df.columns and "trend_price_vs_sma20_atr" in df.columns:
        adv["htf_momentum_agreement"] = np.sign(df["trend_h1_close_vs_sma20_atr"]) * np.sign(
            df["trend_price_vs_sma20_atr"]
        )

    # === Rate of change of key features (5 features) ===
    roc_cols = ["mom_rsi14", "trend_adx14", "cyc_bb_width", "cyc_atr14_normalized", "cyc_zscore20"]
    for col in roc_cols:
        if col in df.columns:
            adv[f"roc3_{col}"] = df[col].pct_change(3)

    # === Additional advanced features to reach ~79 ===
    # Cumulative signal strength
    if all(c in df.columns for c in ["trend_ema9_gt_ema21", "mom_momentum_all_bull", "cyc_bb_pct_b"]):
        adv["signal_strength_bull"] = (
            df["trend_ema9_gt_ema21"]
            + df["mom_momentum_all_bull"]
            + (df["cyc_bb_pct_b"] > 0.5).astype(int)
        )

    if all(c in df.columns for c in ["trend_ema21_gt_ema55", "mom_momentum_all_bear", "cyc_bb_pct_b"]):
        adv["signal_strength_bear"] = (
            (1 - df["trend_ema21_gt_ema55"])
            + df["mom_momentum_all_bear"]
            + (df["cyc_bb_pct_b"] < 0.5).astype(int)
        )

    # RSI regime
    if "mom_rsi14" in df.columns:
        adv["rsi_regime"] = pd.cut(
            df["mom_rsi14"],
            bins=[0, 30, 45, 55, 70, 100],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

    # ATR regime category
    if "cyc_atr_regime_percentile" in df.columns:
        adv["vol_regime"] = pd.cut(
            df["cyc_atr_regime_percentile"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

    # Price acceleration (2nd derivative)
    adv["price_accel"] = close.diff().diff()

    # Volume-price divergence
    if "mom_volume_ratio" in df.columns:
        adv["vol_price_divergence"] = (
            (close > close.shift(1)).astype(int) - (df["mom_volume_ratio"] > 1.0).astype(int)
        )

    # Momentum exhaustion
    if "mom_rsi14" in df.columns and "mom_stoch_k" in df.columns:
        adv["momentum_exhaustion_bull"] = (
            (df["mom_rsi14"] > 70) & (df["mom_stoch_k"] > 80)
        ).astype(int)
        adv["momentum_exhaustion_bear"] = (
            (df["mom_rsi14"] < 30) & (df["mom_stoch_k"] < 20)
        ).astype(int)

    # Trend quality (trend + low volatility = good trend)
    if "trend_adx14" in df.columns and "cyc_atr_regime_percentile" in df.columns:
        adv["trend_quality"] = (df["trend_adx14"] / 100.0) * (1.0 - df["cyc_atr_regime_percentile"])

    # BB position change
    if "cyc_bb_pct_b" in df.columns:
        adv["bb_pct_b_delta3"] = df["cyc_bb_pct_b"].diff(3)
        adv["bb_pct_b_delta5"] = df["cyc_bb_pct_b"].diff(5)

    # MACD histogram momentum
    if "mom_macd_histogram" in df.columns:
        adv["macd_hist_accel"] = df["mom_macd_histogram"].diff().diff()

    # Stochastic divergence proxy
    if "mom_stoch_k" in df.columns and "mom_stoch_d" in df.columns:
        adv["stoch_kd_spread"] = df["mom_stoch_k"] - df["mom_stoch_d"]
        adv["stoch_kd_spread_roc3"] = adv["stoch_kd_spread"].diff(3)

    # Combine everything
    result = pd.concat([df, adv], axis=1)

    # Remove duplicate columns if any
    result = result.loc[:, ~result.columns.duplicated()]

    return result


def create_target(df, horizon=4):
    """
    Create binary target: 1 if close[horizon] > close[0], else 0.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column.
    horizon : int
        Number of bars ahead. Default=4 for M15 = 60-minute prediction.
    
    Returns
    -------
    pd.Series with binary target.
    """
    future_close = df["close"].shift(-horizon)
    target = (future_close > df["close"]).astype(int)
    target.name = "target"
    return target


def prepare_dataset(df_m15, df_h1=None, horizon=4):
    """
    Full pipeline: base features → advanced features → target → clean NaN.
    
    Parameters
    ----------
    df_m15 : pd.DataFrame
        M15 OHLCV DataFrame with DatetimeIndex.
    df_h1 : pd.DataFrame, optional
        H1 OHLCV DataFrame for higher-timeframe features.
    horizon : int
        Target horizon in bars. Default=4 (60 minutes for M15).
    
    Returns
    -------
    X : pd.DataFrame — feature matrix (NaN rows dropped)
    y : pd.Series — binary target aligned with X
    """
    # Step 1: Base features
    df_features = compute_base_features(df_m15, df_h1)

    # Step 2: Advanced features
    df_enhanced = compute_advanced_features(df_features)

    # Step 3: Create target
    y = create_target(df_enhanced, horizon=horizon)
    df_enhanced["target"] = y

    # Step 4: Drop OHLCV columns for ML
    drop_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df_enhanced.columns]
    feature_cols = [c for c in df_enhanced.columns if c not in drop_cols and c != "target"]

    # Step 5: Clean NaN — drop rows with any NaN
    df_clean = df_enhanced.dropna(subset=feature_cols + ["target"])

    X = df_clean[feature_cols]
    y = df_clean["target"]

    return X, y


# ---------------------------------------------------------------------------
# Main entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("feature_engineering_v2.py — 3-Concept Framework")
    print("=" * 55)
    print(f"Trend features:   33")
    print(f"Momentum features: 33")
    print(f"Cycle features:    33")
    print(f"Base total:        99")
    print(f"Advanced features: ~79")
    print(f"Grand total:       ~178")
    print()
    print("Use compute_base_features(df_m15, df_h1) to generate features.")
    print("Use compute_advanced_features(df_features) to enhance them.")
    print("Use prepare_dataset(df_m15, df_h1) for the full pipeline.")
