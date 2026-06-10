#!/usr/bin/env python3
"""
XAUUSD Technical Analysis Module
=================================

Comprehensive multi-indicator technical analysis for XAUUSD (Gold),
adapted for gold's specific characteristics:
  - Higher volatility → wider ATR ranges (5-15 on M15, 15-40 on H1)
  - Fibonacci levels are critical (38.2%, 50%, 61.8%)
  - Session dynamics: London Fix (15:00 UTC) and COMEX open are key
  - Default stop sizing: 1.5x ATR (gold needs wider stops than forex)

Uses only pandas and numpy — no ta-lib dependency.
Optimised for 45K+ rows of M15 data.

Functions
---------
calculate_indicators(df, timeframe='H1')
    Add SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR,
    Fibonacci retracement, and Volume SMA columns.

get_technical_signals(df)
    Return a dict of current signal states for the latest bar.

calculate_confluence_score(signals, direction)
    Compute a confluence-based confidence score (0-10).

get_session_context(timestamp_utc)
    Return trading session info for a UTC timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# ── Gold-specific constants ──────────────────────────────────────────────────

# Typical ATR ranges for XAUUSD by timeframe (used for sanity checks / docs)
GOLD_ATR_RANGES = {
    "M15": (5, 15),
    "H1":  (15, 40),
    "H4":  (30, 80),
    "D1":  (60, 200),
}

# Default ATR multiplier for stop-loss sizing (gold needs wider stops)
ATR_STOP_MULTIPLIER = 1.5

# RSI thresholds — gold can stay overbought/oversold longer than FX
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Stochastic thresholds
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20

# Bollinger Band parameters
BB_PERIOD = 20
BB_STD = 2.0

# Fibonacci retracement levels (gold respects 38.2%, 50%, 61.8% the most)
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
# Extra emphasis levels for gold
FIB_KEY_LEVELS = [0.382, 0.5, 0.618]

# ── Session boundaries (UTC hours) ──────────────────────────────────────────
# Gold trades nearly 23h/day but liquidity/volatility is session-dependent.
# Key events: London Fix (15:00 UTC), COMEX open (~13:30 UTC for pre-market)

SESSIONS = {
    "ASIAN":           (0, 7),       # Tokyo / Sydney
    "LONDON":          (7, 12),      # London open
    "LONDON_NY_OVERLAP": (12, 17),   # Peak gold hours — London + NY overlap
    "NY":              (17, 21),     # NY afternoon (post-London)
    "OFF_HOURS":       (21, 24),     # Low liquidity
}

# Session volatility profile for gold (relative)
SESSION_VOLATILITY = {
    "ASIAN":             "LOW",
    "LONDON":            "MEDIUM",
    "LONDON_NY_OVERLAP": "HIGH",
    "NY":                "MEDIUM",
    "OFF_HOURS":         "LOW",
}

# Key gold event times (UTC)
LONDON_FIX_UTC = 15       # 15:00 UTC — LBMA Gold Price PM Fix
COMEX_OPEN_UTC = 13       # ~13:30 UTC COMEX pre-market; 13:00 label
COMEX_OPEN_MIN = 30       # 13:30 UTC actual open


# ══════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL INDICATOR CALCULATIONS (vectorised with pandas/numpy)
# ══════════════════════════════════════════════════════════════════════════════

def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing method).
    Gold can stay overbought/oversold longer than FX pairs.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (exponential with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD indicator.
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    series: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns (upper_band, middle_band, lower_band).
    """
    middle = _sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    Returns (%K, %D).
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(window=k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(window=d_smooth, min_periods=d_smooth).mean()
    return k, d


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range — critical for gold's wider ranges.
    Uses Wilder's smoothing (same as RSI).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _find_swing_points(
    high: pd.Series,
    low: pd.Series,
    window: int = 10,
) -> Tuple[pd.Series, pd.Series]:
    """
    Identify local swing highs and swing lows over a rolling window.
    A swing high at index i means high[i] == max(high[i-window:i+window+1]).
    Returns boolean Series for swing_high and swing_low.
    """
    roll_high = high.rolling(window=2 * window + 1, center=True, min_periods=window).max()
    roll_low = low.rolling(window=2 * window + 1, center=True, min_periods=window).min()
    swing_high = high == roll_high
    swing_low = low == roll_low
    return swing_high, swing_low


def _fibonacci_retracement(
    high: pd.Series,
    low: pd.Series,
    window: int = 50,
) -> pd.DataFrame:
    """
    Calculate Fibonacci retracement levels from the most recent swing high/low
    within the given lookback window.

    Returns a DataFrame with columns: fib_0.0, fib_0.236, fib_0.382,
    fib_0.5, fib_0.618, fib_0.786, fib_1.0
    """
    swing_high_flag, swing_low_flag = _find_swing_points(high, low, window=min(10, window // 3))

    # Initialise output
    fib_df = pd.DataFrame(index=high.index, dtype=float)

    # We compute rolling latest swing high and swing low
    # For efficiency, we track the most-recent swing price going forward
    latest_swing_high = high.copy()
    latest_swing_low = low.copy()

    # Simple forward-fill approach: mark swing points, then ffill
    sh_values = high.where(swing_high_flag)
    sl_values = low.where(swing_low_flag)
    sh_values = sh_values.ffill()
    sl_values = sl_values.ffill()

    # For the first rows where no swing has been found yet, use rolling max/min
    sh_values = sh_values.fillna(high.rolling(window, min_periods=1).max())
    sl_values = sl_values.fillna(low.rolling(window, min_periods=1).min())

    diff = sh_values - sl_values

    for level in FIB_LEVELS:
        col = f"fib_{level:.3f}"
        fib_df[col] = sh_values - diff * level

    return fib_df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame, timeframe: str = "H1") -> pd.DataFrame:
    """
    Add technical indicator columns to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
        May have a datetime index (recommended).
    timeframe : str
        One of 'M15', 'H1', 'H4', 'D1'. Used for gold-specific adjustments.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the following new columns:

        SMA columns:    sma_20, sma_50, sma_200
        EMA columns:    ema_9, ema_21, ema_55
        RSI column:     rsi_14
        MACD columns:   macd_line, macd_signal, macd_histogram
        BB columns:     bb_upper, bb_middle, bb_lower
        Stoch columns:  stoch_k, stoch_d
        ATR column:     atr_14
        Fib columns:    fib_0.000 .. fib_1.000
        Volume column:  volume_sma_20
    """
    df = df.copy()

    # ── SMA ───────────────────────────────────────────────────────────────
    df["sma_20"]  = _sma(df["close"], 20)
    df["sma_50"]  = _sma(df["close"], 50)
    df["sma_200"] = _sma(df["close"], 200)

    # ── EMA ───────────────────────────────────────────────────────────────
    df["ema_9"]   = _ema(df["close"], 9)
    df["ema_21"]  = _ema(df["close"], 21)
    df["ema_55"]  = _ema(df["close"], 55)

    # ── RSI ───────────────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(df["close"], 14)

    # ── MACD ──────────────────────────────────────────────────────────────
    macd_line, macd_signal, macd_histogram = _macd(df["close"], 12, 26, 9)
    df["macd_line"]     = macd_line
    df["macd_signal"]   = macd_signal
    df["macd_histogram"] = macd_histogram

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_upper, bb_middle, bb_lower = _bollinger_bands(df["close"])
    df["bb_upper"]  = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"]  = bb_lower

    # ── Stochastic ────────────────────────────────────────────────────────
    stoch_k, stoch_d = _stochastic(df["high"], df["low"], df["close"], 14, 3, 3)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    # ── ATR ───────────────────────────────────────────────────────────────
    df["atr_14"] = _atr(df["high"], df["low"], df["close"], 14)

    # ── Fibonacci Retracement ─────────────────────────────────────────────
    # Use a lookback window appropriate for the timeframe
    fib_window = {"M15": 100, "H1": 50, "H4": 30, "D1": 20}.get(timeframe, 50)
    fib_df = _fibonacci_retracement(df["high"], df["low"], window=fib_window)
    for col in fib_df.columns:
        df[col] = fib_df[col]

    # ── Volume SMA ────────────────────────────────────────────────────────
    df["volume_sma_20"] = _sma(df["volume"].astype(float), 20)

    return df


def get_technical_signals(df: pd.DataFrame) -> Dict:
    """
    Evaluate the latest bar's indicator readings and return a dict of
    current signal states.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that already has indicator columns (i.e. after
        calling ``calculate_indicators``).

    Returns
    -------
    dict
        Keys and their possible values:

        ema_cross     : BULLISH | BEARISH | NEUTRAL
        rsi_signal    : OVERBOUGHT | OVERSOLD | NEUTRAL
        rsi_value     : float
        rsi_divergence: BULLISH | BEARISH | NONE
        macd_signal   : BULLISH | BEARISH | NEUTRAL
        macd_divergence: BULLISH | BEARISH | NONE
        bb_signal     : OVERBOUGHT | OVERSOLD | NEUTRAL
        bb_position   : float  (0 = lower band, 1 = upper band)
        stoch_signal  : OVERBOUGHT | OVERSOLD | NEUTRAL
        trend         : UPTREND | DOWNTREND | SIDEWAYS
        atr_value     : float
        fib_levels    : dict of {level_name: price}
        price         : float (latest close)
        sma_20        : float
        sma_50        : float
        sma_200       : float
    """
    # Use the last row for current state
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    signals: Dict = {}

    # ── Price ─────────────────────────────────────────────────────────────
    signals["price"] = float(last["close"])

    # ── EMA Cross ─────────────────────────────────────────────────────────
    ema9_now = last.get("ema_9")
    ema21_now = last.get("ema_21")
    ema9_prev = prev.get("ema_9")
    ema21_prev = prev.get("ema_21")

    if pd.notna(ema9_now) and pd.notna(ema21_now):
        if ema9_now > ema21_now:
            # Check if a fresh cross just happened
            if pd.notna(ema9_prev) and pd.notna(ema21_prev) and ema9_prev <= ema21_prev:
                signals["ema_cross"] = "BULLISH"  # fresh cross up
            else:
                signals["ema_cross"] = "BULLISH"
        elif ema9_now < ema21_now:
            if pd.notna(ema9_prev) and pd.notna(ema21_prev) and ema9_prev >= ema21_prev:
                signals["ema_cross"] = "BEARISH"  # fresh cross down
            else:
                signals["ema_cross"] = "BEARISH"
        else:
            signals["ema_cross"] = "NEUTRAL"
    else:
        signals["ema_cross"] = "NEUTRAL"

    # ── RSI ───────────────────────────────────────────────────────────────
    rsi_val = last.get("rsi_14", np.nan)
    signals["rsi_value"] = float(rsi_val) if pd.notna(rsi_val) else None

    if pd.notna(rsi_val):
        if rsi_val >= RSI_OVERBOUGHT:
            signals["rsi_signal"] = "OVERBOUGHT"
        elif rsi_val <= RSI_OVERSOLD:
            signals["rsi_signal"] = "OVERSOLD"
        else:
            signals["rsi_signal"] = "NEUTRAL"
    else:
        signals["rsi_signal"] = "NEUTRAL"

    # RSI Divergence detection (look back ~20 bars)
    signals["rsi_divergence"] = _detect_rsi_divergence(df)

    # ── MACD ──────────────────────────────────────────────────────────────
    macd_line_now = last.get("macd_line")
    macd_sig_now = last.get("macd_signal")
    macd_hist_now = last.get("macd_histogram")
    macd_line_prev = prev.get("macd_line")
    macd_sig_prev = prev.get("macd_signal")

    if pd.notna(macd_line_now) and pd.notna(macd_sig_now):
        # Crossover on current bar
        fresh_bullish_cross = (
            pd.notna(macd_line_prev) and pd.notna(macd_sig_prev)
            and macd_line_prev <= macd_sig_prev
            and macd_line_now > macd_sig_now
        )
        fresh_bearish_cross = (
            pd.notna(macd_line_prev) and pd.notna(macd_sig_prev)
            and macd_line_prev >= macd_sig_prev
            and macd_line_now < macd_sig_now
        )

        if fresh_bullish_cross:
            signals["macd_signal"] = "BULLISH"
        elif fresh_bearish_cross:
            signals["macd_signal"] = "BEARISH"
        elif macd_line_now > macd_sig_now and pd.notna(macd_hist_now) and macd_hist_now > 0:
            signals["macd_signal"] = "BULLISH"  # above signal, positive histogram
        elif macd_line_now < macd_sig_now and pd.notna(macd_hist_now) and macd_hist_now < 0:
            signals["macd_signal"] = "BEARISH"
        else:
            signals["macd_signal"] = "NEUTRAL"
    else:
        signals["macd_signal"] = "NEUTRAL"

    # MACD Divergence detection
    signals["macd_divergence"] = _detect_macd_divergence(df)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_upper = last.get("bb_upper")
    bb_lower = last.get("bb_lower")
    bb_middle = last.get("bb_middle")
    price = last["close"]

    if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_middle):
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_position = (price - bb_lower) / bb_range
        else:
            bb_position = 0.5
        signals["bb_position"] = float(bb_position)

        if price >= bb_upper:
            signals["bb_signal"] = "OVERBOUGHT"
        elif price <= bb_lower:
            signals["bb_signal"] = "OVERSOLD"
        else:
            signals["bb_signal"] = "NEUTRAL"
    else:
        signals["bb_signal"] = "NEUTRAL"
        signals["bb_position"] = None

    # ── Stochastic ────────────────────────────────────────────────────────
    stoch_k_val = last.get("stoch_k")
    stoch_d_val = last.get("stoch_d")

    if pd.notna(stoch_k_val) and pd.notna(stoch_d_val):
        if stoch_k_val >= STOCH_OVERBOUGHT and stoch_d_val >= STOCH_OVERBOUGHT:
            signals["stoch_signal"] = "OVERBOUGHT"
        elif stoch_k_val <= STOCH_OVERSOLD and stoch_d_val <= STOCH_OVERSOLD:
            signals["stoch_signal"] = "OVERSOLD"
        else:
            signals["stoch_signal"] = "NEUTRAL"
    else:
        signals["stoch_signal"] = "NEUTRAL"

    # ── Trend (SMA alignment) ────────────────────────────────────────────
    sma20 = last.get("sma_20")
    sma50 = last.get("sma_50")
    sma200 = last.get("sma_200")
    signals["sma_20"] = float(sma20) if pd.notna(sma20) else None
    signals["sma_50"] = float(sma50) if pd.notna(sma50) else None
    signals["sma_200"] = float(sma200) if pd.notna(sma200) else None

    if pd.notna(sma20) and pd.notna(sma50) and pd.notna(sma200):
        if sma20 > sma50 > sma200:
            signals["trend"] = "UPTREND"
        elif sma20 < sma50 < sma200:
            signals["trend"] = "DOWNTREND"
        else:
            signals["trend"] = "SIDEWAYS"
    else:
        signals["trend"] = "SIDEWAYS"

    # ── ATR ───────────────────────────────────────────────────────────────
    atr_val = last.get("atr_14")
    signals["atr_value"] = float(atr_val) if pd.notna(atr_val) else None

    # ── Fibonacci levels ──────────────────────────────────────────────────
    fib_levels = {}
    for level in FIB_LEVELS:
        col = f"fib_{level:.3f}"
        val = last.get(col)
        if pd.notna(val):
            fib_levels[f"Fib {level*100:.1f}%"] = float(val)
    signals["fib_levels"] = fib_levels

    return signals


# ── Divergence detection helpers ──────────────────────────────────────────

def _detect_rsi_divergence(
    df: pd.DataFrame,
    lookback: int = 20,
) -> str:
    """
    Detect RSI divergence over the last *lookback* bars.

    Bullish divergence : price makes lower low, RSI makes higher low
    Bearish divergence : price makes higher high, RSI makes lower high

    Returns 'BULLISH', 'BEARISH', or 'NONE'.
    """
    if len(df) < lookback + 1:
        return "NONE"

    recent = df.iloc[-lookback:]
    rsi_col = "rsi_14"
    if rsi_col not in recent.columns:
        return "NONE"

    # Drop NaNs
    valid = recent[["close", rsi_col]].dropna()
    if len(valid) < 5:
        return "NONE"

    # Find local price and RSI extremes (simple: min and max positions)
    price_min_idx = valid["close"].idxmin()
    price_max_idx = valid["close"].idxmax()
    rsi_at_price_min = valid.loc[price_min_idx, rsi_col]
    rsi_at_price_max = valid.loc[price_max_idx, rsi_col]

    # Split into two halves for divergence comparison
    mid = len(valid) // 2
    if mid < 2:
        return "NONE"

    first_half = valid.iloc[:mid]
    second_half = valid.iloc[mid:]

    # Bullish divergence: price lower low in second half, RSI higher low
    price_ll_first = first_half["close"].min()
    price_ll_second = second_half["close"].min()
    rsi_ll_first = first_half[rsi_col].min()
    rsi_ll_second = second_half[rsi_col].min()

    if price_ll_second < price_ll_first and rsi_ll_second > rsi_ll_first:
        return "BULLISH"

    # Bearish divergence: price higher high in second half, RSI lower high
    price_hh_first = first_half["close"].max()
    price_hh_second = second_half["close"].max()
    rsi_hh_first = first_half[rsi_col].max()
    rsi_hh_second = second_half[rsi_col].max()

    if price_hh_second > price_hh_first and rsi_hh_second < rsi_hh_first:
        return "BEARISH"

    return "NONE"


def _detect_macd_divergence(
    df: pd.DataFrame,
    lookback: int = 25,
) -> str:
    """
    Detect MACD divergence over the last *lookback* bars.

    Bullish divergence : price makes lower low, MACD histogram makes higher low
    Bearish divergence : price makes higher high, MACD histogram makes lower high

    Returns 'BULLISH', 'BEARISH', or 'NONE'.
    """
    if len(df) < lookback + 1:
        return "NONE"

    recent = df.iloc[-lookback:]
    hist_col = "macd_histogram"
    if hist_col not in recent.columns:
        return "NONE"

    valid = recent[["close", hist_col]].dropna()
    if len(valid) < 5:
        return "NONE"

    mid = len(valid) // 2
    if mid < 2:
        return "NONE"

    first_half = valid.iloc[:mid]
    second_half = valid.iloc[mid:]

    # Bullish divergence
    if (second_half["close"].min() < first_half["close"].min()
            and second_half[hist_col].min() > first_half[hist_col].min()):
        return "BULLISH"

    # Bearish divergence
    if (second_half["close"].max() > first_half["close"].max()
            and second_half[hist_col].max() < first_half[hist_col].max()):
        return "BEARISH"

    return "NONE"


# ══════════════════════════════════════════════════════════════════════════════
#  CONFLUENCE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_confluence_score(signals: Dict, direction: str) -> float:
    """
    Calculate a confluence-based confidence score (0-10) for a given direction.

    Parameters
    ----------
    signals : dict
        Output of ``get_technical_signals``.
    direction : str
        'BULLISH' or 'BEARISH'.

    Returns
    -------
    float
        Score from 0 to 10 (capped).

    Scoring weights (total possible > 10; score is capped at 10):
    ─────────────────────────────────────────────
    Factor                                      Weight
    ─────────────────────────────────────────────
    EMA 9/21 crossover in direction              1.5
    RSI supports direction                       1.0
    RSI divergence in direction                  1.5
    MACD crossover confirms                      1.2
    MACD divergence in direction                 1.5
    Price at Bollinger Band extreme              0.8
    Fibonacci level at entry zone                1.0
    Stochastic extreme + crossover               0.7
    Higher timeframe trend alignment             1.0
    Volume increasing                            0.5
    Key S/R level nearby                         0.5
    ─────────────────────────────────────────────
    """
    if direction not in ("BULLISH", "BEARISH"):
        raise ValueError("direction must be 'BULLISH' or 'BEARISH'")

    score = 0.0
    opposite = "BEARISH" if direction == "BULLISH" else "BULLISH"

    # ── EMA 9/21 crossover in direction: +1.5 ────────────────────────────
    if signals.get("ema_cross") == direction:
        score += 1.5

    # ── RSI supports direction: +1.0 ─────────────────────────────────────
    rsi_sig = signals.get("rsi_signal")
    if direction == "BULLISH" and rsi_sig == "OVERSOLD":
        score += 1.0
    elif direction == "BEARISH" and rsi_sig == "OVERBOUGHT":
        score += 1.0

    # ── RSI divergence: +1.5 ─────────────────────────────────────────────
    if signals.get("rsi_divergence") == direction:
        score += 1.5

    # ── MACD crossover confirms: +1.2 ────────────────────────────────────
    if signals.get("macd_signal") == direction:
        score += 1.2

    # ── MACD divergence: +1.5 ────────────────────────────────────────────
    if signals.get("macd_divergence") == direction:
        score += 1.5

    # ── Price at Bollinger Band extreme: +0.8 ────────────────────────────
    bb_sig = signals.get("bb_signal")
    if direction == "BULLISH" and bb_sig == "OVERSOLD":
        score += 0.8
    elif direction == "BEARISH" and bb_sig == "OVERBOUGHT":
        score += 0.8

    # ── Fibonacci level at entry zone: +1.0 ──────────────────────────────
    # Check if price is near a key Fibonacci level (within 0.3% for gold)
    fib_levels = signals.get("fib_levels", {})
    price = signals.get("price")
    if price and fib_levels:
        atr_val = signals.get("atr_value") or 0
        fib_tolerance = max(atr_val * 0.5, price * 0.003)  # 0.3% or 0.5*ATR
        near_key_fib = False
        for fib_name, fib_price in fib_levels.items():
            # Check if it's a key Fibonacci level
            is_key = any(f"{lvl*100:.1f}%" in fib_name for lvl in FIB_KEY_LEVELS)
            if is_key and abs(price - fib_price) <= fib_tolerance:
                near_key_fib = True
                break
        if near_key_fib:
            score += 1.0

    # ── Stochastic extreme + crossover: +0.7 ─────────────────────────────
    stoch_sig = signals.get("stoch_signal")
    if direction == "BULLISH" and stoch_sig == "OVERSOLD":
        score += 0.7
    elif direction == "BEARISH" and stoch_sig == "OVERBOUGHT":
        score += 0.7

    # ── Higher timeframe trend alignment: +1.0 ───────────────────────────
    trend = signals.get("trend")
    if direction == "BULLISH" and trend == "UPTREND":
        score += 1.0
    elif direction == "BEARISH" and trend == "DOWNTREND":
        score += 1.0

    # ── Volume increasing: +0.5 ──────────────────────────────────────────
    # We check this from the signals dict if volume info is available
    # (caller should add 'volume_increasing' key if they have volume data)
    if signals.get("volume_increasing"):
        score += 0.5

    # ── Key S/R level nearby: +0.5 ───────────────────────────────────────
    # (caller should add 'sr_nearby' key if they have S/R data)
    if signals.get("sr_nearby"):
        score += 0.5

    # Cap at 10
    return min(round(score, 2), 10.0)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def get_session_context(timestamp_utc) -> Dict:
    """
    Return trading session info for a given UTC timestamp.

    Parameters
    ----------
    timestamp_utc : datetime-like
        A UTC timestamp (str, pd.Timestamp, or datetime).

    Returns
    -------
    dict
        current_session     : str  — ASIAN / LONDON / NY / LONDON_NY_OVERLAP / OFF_HOURS
        next_session        : str  — name of the next session
        time_to_next_session: str  — human-readable time until next session
        next_session_utc    : datetime — UTC start of next session
        session_volatility  : str  — LOW / MEDIUM / HIGH
        is_london_fix       : bool — within 30 min of London Fix (15:00 UTC)
        is_comex_open       : bool — within 30 min of COMEX open (13:30 UTC)
        gold_event          : str or None — name of an active gold-specific event
    """
    if isinstance(timestamp_utc, str):
        ts = pd.Timestamp(timestamp_utc, tz="UTC")
    elif isinstance(timestamp_utc, pd.Timestamp):
        ts = timestamp_utc
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
    elif isinstance(timestamp_utc, datetime):
        ts = pd.Timestamp(timestamp_utc, tz="UTC")
    else:
        ts = pd.Timestamp(timestamp_utc, tz="UTC")

    hour = ts.hour
    minute = ts.minute

    # ── Determine current session ─────────────────────────────────────────
    current_session = "OFF_HOURS"
    for name, (start_h, end_h) in SESSIONS.items():
        if start_h <= hour < end_h:
            current_session = name
            break

    # ── Session volatility ────────────────────────────────────────────────
    session_volatility = SESSION_VOLATILITY.get(current_session, "LOW")

    # ── London Fix & COMEX open ───────────────────────────────────────────
    is_london_fix = False
    is_comex_open = False
    gold_event = None

    # London Fix at 15:00 UTC — active within ±30 min
    if (hour == 14 and minute >= 30) or (hour == 15 and minute <= 30):
        is_london_fix = True
        gold_event = "London Fix"

    # COMEX open at 13:30 UTC — active within ±30 min
    if (hour == 13 and minute >= 0) or (hour == 14 and minute <= 0):
        is_comex_open = True
        if gold_event is None:
            gold_event = "COMEX Open"

    # ── Next session ──────────────────────────────────────────────────────
    session_order = ["ASIAN", "LONDON", "LONDON_NY_OVERLAP", "NY", "OFF_HOURS"]
    try:
        current_idx = session_order.index(current_session)
    except ValueError:
        current_idx = 4  # OFF_HOURS

    next_session_name = session_order[(current_idx + 1) % len(session_order)]
    next_session_start_hour = SESSIONS[next_session_name][0]

    # Calculate time to next session
    today = ts.normalize()
    next_start = today + timedelta(hours=next_session_start_hour)

    # If the next session's start hour is <= current hour, it starts tomorrow
    if next_start <= ts:
        next_start = today + timedelta(days=1, hours=next_session_start_hour)

    time_delta = next_start - ts
    hours_left = int(time_delta.total_seconds() // 3600)
    minutes_left = int((time_delta.total_seconds() % 3600) // 60)
    time_to_next = f"{hours_left}h {minutes_left}m"

    return {
        "current_session": current_session,
        "next_session": next_session_name,
        "time_to_next_session": time_to_next,
        "next_session_utc": next_start,
        "session_volatility": session_volatility,
        "is_london_fix": is_london_fix,
        "is_comex_open": is_comex_open,
        "gold_event": gold_event,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GOLD-SPECIFIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_gold_stop_loss(
    atr_value: float,
    direction: str,
    entry_price: float,
    multiplier: float = ATR_STOP_MULTIPLIER,
    min_stop_points: float = 5.0,
) -> Dict:
    """
    Calculate gold-specific stop-loss and take-profit levels.

    Gold needs wider stops than forex. Default is 1.5x ATR.

    Parameters
    ----------
    atr_value : float
        Current ATR(14) value.
    direction : str
        'BULLISH' or 'BEARISH'.
    entry_price : float
        Entry price.
    multiplier : float
        ATR multiplier for stop distance (default 1.5).
    min_stop_points : float
        Minimum stop distance in points (safety floor).

    Returns
    -------
    dict
        stop_loss     : float
        take_profit_1 : float  (1:1 R:R)
        take_profit_2 : float  (1:2 R:R)
        take_profit_3 : float  (1:3 R:R)
        stop_distance : float
    """
    stop_distance = max(atr_value * multiplier, min_stop_points)

    if direction == "BULLISH":
        stop_loss = entry_price - stop_distance
        take_profit_1 = entry_price + stop_distance      # 1:1
        take_profit_2 = entry_price + stop_distance * 2  # 1:2
        take_profit_3 = entry_price + stop_distance * 3  # 1:3
    else:
        stop_loss = entry_price + stop_distance
        take_profit_1 = entry_price - stop_distance
        take_profit_2 = entry_price - stop_distance * 2
        take_profit_3 = entry_price - stop_distance * 3

    return {
        "stop_loss": round(stop_loss, 2),
        "take_profit_1": round(take_profit_1, 2),
        "take_profit_2": round(take_profit_2, 2),
        "take_profit_3": round(take_profit_3, 2),
        "stop_distance": round(stop_distance, 2),
    }


def enrich_signals_with_volume(df: pd.DataFrame, signals: Dict) -> Dict:
    """
    Add volume-related signal data to the signals dict.

    Sets 'volume_increasing' = True if current volume > volume_sma_20.
    This is used by ``calculate_confluence_score``.
    """
    if len(df) < 2:
        return signals

    last = df.iloc[-1]
    vol_now = last.get("volume")
    vol_sma = last.get("volume_sma_20")

    if pd.notna(vol_now) and pd.notna(vol_sma):
        signals["volume_increasing"] = bool(vol_now > vol_sma)
    else:
        signals["volume_increasing"] = False

    return signals


def enrich_signals_with_sr(df: pd.DataFrame, signals: Dict, tolerance_atr_ratio: float = 0.3) -> Dict:
    """
    Add support/resistance proximity signal.

    Checks if price is near a recent swing high or swing low (potential S/R).
    Uses ATR for dynamic tolerance.

    Sets 'sr_nearby' = True if price is within tolerance_atr_ratio * ATR
    of a recent swing point.
    """
    if len(df) < 50:
        signals["sr_nearby"] = False
        return signals

    price = signals.get("price")
    atr_val = signals.get("atr_value")

    if price is None or atr_val is None:
        signals["sr_nearby"] = False
        return signals

    tolerance = atr_val * tolerance_atr_ratio

    # Check recent swing highs and lows (last 50 bars)
    recent = df.iloc[-50:]
    swing_h, swing_l = _find_swing_points(recent["high"], recent["low"], window=5)

    swing_high_prices = recent.loc[swing_h, "high"]
    swing_low_prices = recent.loc[swing_l, "low"]

    near_high = any(abs(swing_high_prices - price) <= tolerance)
    near_low = any(abs(swing_low_prices - price) <= tolerance)

    signals["sr_nearby"] = near_high or near_low
    return signals


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: FULL ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    df: pd.DataFrame,
    timeframe: str = "H1",
    direction: str = "BULLISH",
) -> Dict:
    """
    Run the complete technical analysis pipeline on a DataFrame.

    1. Calculate indicators
    2. Get technical signals
    3. Enrich with volume and S/R
    4. Calculate confluence score
    5. Get session context (if timestamp available)
    6. Calculate stop-loss / take-profit levels

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (open, high, low, close, volume).
    timeframe : str
        'M15', 'H1', 'H4', or 'D1'.
    direction : str
        'BULLISH' or 'BEARISH'.

    Returns
    -------
    dict
        Complete analysis result including signals, score, session, and stops.
    """
    # Step 1: Calculate indicators
    df = calculate_indicators(df, timeframe=timeframe)

    # Step 2: Get signals
    signals = get_technical_signals(df)

    # Step 3: Enrich with volume & S/R
    signals = enrich_signals_with_volume(df, signals)
    signals = enrich_signals_with_sr(df, signals)

    # Step 4: Confluence score
    confluence_score = calculate_confluence_score(signals, direction)

    # Step 5: Session context
    session_ctx = {}
    if isinstance(df.index[-1], pd.Timestamp):
        session_ctx = get_session_context(df.index[-1])

    # Step 6: Stop-loss / take-profit
    stops = {}
    atr_val = signals.get("atr_value")
    price = signals.get("price")
    if atr_val and price:
        stops = calculate_gold_stop_loss(atr_val, direction, price)

    return {
        "timeframe": timeframe,
        "direction": direction,
        "signals": signals,
        "confluence_score": confluence_score,
        "session_context": session_ctx,
        "stops": stops,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT (for quick testing)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import json

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # Default to H1 for quick demo
    tf = "H1"
    csv_path = os.path.join(DATA_DIR, f"XAUUSD_{tf}.csv")

    if not os.path.exists(csv_path):
        print(f"[ERROR] Data file not found: {csv_path}")
        print("Run aggregate_data.py first to generate timeframe CSVs.")
        raise SystemExit(1)

    print(f"[INFO] Loading {tf} data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    print(f"[INFO] Loaded {len(df):,} rows  |  "
          f"Date range: {df.index.min()} → {df.index.max()}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Use only the last 500 rows for a quick demo
    df_tail = df.tail(500).copy()
    print(f"[INFO] Running analysis on last {len(df_tail)} bars...")

    for direction in ("BULLISH", "BEARISH"):
        result = run_full_analysis(df_tail, timeframe=tf, direction=direction)

        print(f"\n{'='*70}")
        print(f"  XAUUSD {tf} — {direction} Analysis")
        print(f"{'='*70}")
        print(f"  Price          : {result['signals']['price']:.2f}")
        print(f"  Trend          : {result['signals']['trend']}")
        print(f"  EMA Cross      : {result['signals']['ema_cross']}")
        print(f"  RSI            : {result['signals']['rsi_value']:.1f} "
              f"({result['signals']['rsi_signal']})")
        print(f"  RSI Divergence : {result['signals']['rsi_divergence']}")
        print(f"  MACD Signal    : {result['signals']['macd_signal']}")
        print(f"  MACD Divergence: {result['signals']['macd_divergence']}")
        print(f"  BB Signal      : {result['signals']['bb_signal']} "
              f"(position: {result['signals']['bb_position']:.2f})")
        print(f"  Stochastic     : {result['signals']['stoch_signal']}")
        print(f"  ATR(14)        : {result['signals']['atr_value']:.2f}")
        print(f"  Fib Levels     : {json.dumps(result['signals']['fib_levels'], indent=4)}")
        print(f"  Vol Increasing : {result['signals']['volume_increasing']}")
        print(f"  S/R Nearby     : {result['signals']['sr_nearby']}")
        print(f"  ────────────────────────────────────")
        print(f"  Confluence Score: {result['confluence_score']:.1f} / 10")

        if result["stops"]:
            print(f"  Stop Loss      : {result['stops']['stop_loss']:.2f}")
            print(f"  TP1 (1:1)      : {result['stops']['take_profit_1']:.2f}")
            print(f"  TP2 (1:2)      : {result['stops']['take_profit_2']:.2f}")
            print(f"  TP3 (1:3)      : {result['stops']['take_profit_3']:.2f}")
            print(f"  Stop Distance  : {result['stops']['stop_distance']:.2f}")

        if result["session_context"]:
            sc = result["session_context"]
            print(f"  ────────────────────────────────────")
            print(f"  Session        : {sc['current_session']}")
            print(f"  Volatility     : {sc['session_volatility']}")
            print(f"  London Fix     : {sc['is_london_fix']}")
            print(f"  COMEX Open     : {sc['is_comex_open']}")
            if sc["gold_event"]:
                print(f"  Gold Event     : {sc['gold_event']}")
            print(f"  Next Session   : {sc['next_session']} in {sc['time_to_next_session']}")

    print(f"\n{'='*70}")
    print("  Analysis complete.")
    print(f"{'='*70}")
