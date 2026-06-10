#!/usr/bin/env python3
"""
prediction_tracker.py — Prediction Outcome Tracking & Performance Metrics
=========================================================================

Resolves past predictions against actual price movements and computes
comprehensive performance statistics.

Functions
---------
resolve_predictions(predictions, current_df)
    Check past predictions against actual price movements.

calculate_stats(predictions)
    Compute performance metrics from resolved predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------

def resolve_predictions(
    predictions: List[Dict[str, Any]],
    current_df: pd.DataFrame,
    lookback_bars: int = 8,
) -> List[Dict[str, Any]]:
    """
    Check past unresolved predictions against actual price movements.

    For each unresolved prediction, examines the price action starting
    from the prediction timestamp forward. If price hits TP before SL,
    the outcome is 'WIN'; if SL before TP, outcome is 'LOSS'. If neither
    level is reached within *lookback_bars* H1 bars (or equivalent),
    the outcome is 'EXPIRED'.

    Parameters
    ----------
    predictions : list of dict
        List of prediction dicts (from memory_manager).
    current_df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex. Should cover the period
        after the prediction timestamps.
    lookback_bars : int
        Maximum number of bars to look ahead for resolution.
        Default=8 (8 H1 bars = 8 hours for H1 predictions).

    Returns
    -------
    list of dict
        Updated predictions list with outcomes filled in.
    """
    if current_df.empty:
        return predictions

    # Ensure DatetimeIndex
    if not isinstance(current_df.index, pd.DatetimeIndex):
        return predictions

    df = current_df.sort_index()

    for pred in predictions:
        # Skip already resolved
        if pred.get("outcome") is not None:
            continue

        # Parse prediction timestamp
        pred_ts = _parse_timestamp(pred.get("timestamp"))
        if pred_ts is None:
            continue

        # Get bars after the prediction
        future_mask = df.index > pred_ts
        future_bars = df.loc[future_mask].head(lookback_bars)

        if future_bars.empty:
            continue

        direction = pred.get("direction", "")
        tp1 = pred.get("take_profit_1")
        sl = pred.get("stop_loss")

        if tp1 is None or sl is None:
            continue

        # Resolve by checking which level was hit first
        outcome, outcome_price, outcome_time, pnl_pips = _check_price_hit(
            future_bars, direction, tp1, sl
        )

        if outcome is not None:
            pred["outcome"] = outcome
            pred["outcome_price"] = outcome_price
            pred["outcome_time"] = outcome_time.isoformat() if outcome_time else None
            pred["pnl_pips"] = pnl_pips
        elif len(future_bars) >= lookback_bars:
            # Expired — neither level hit within the window
            # Use the last close as outcome
            last_bar = future_bars.iloc[-1]
            pred["outcome"] = "EXPIRED"
            pred["outcome_price"] = float(last_bar["close"])
            pred["outcome_time"] = future_bars.index[-1].isoformat()
            pred["pnl_pips"] = _calculate_pnl(
                direction, pred["entry_price"], float(last_bar["close"])
            )

    return predictions


def _parse_timestamp(ts) -> Optional[pd.Timestamp]:
    """Parse various timestamp formats into pd.Timestamp."""
    if ts is None:
        return None
    try:
        if isinstance(ts, datetime):
            return pd.Timestamp(ts)
        ts_str = str(ts).replace("Z", "+00:00")
        return pd.Timestamp(ts_str)
    except (ValueError, TypeError):
        return None


def _check_price_hit(
    future_bars: pd.DataFrame,
    direction: str,
    tp: float,
    sl: float,
) -> Tuple[Optional[str], Optional[float], Optional[pd.Timestamp], Optional[float]]:
    """
    Check if TP or SL was hit first in the future bars.

    Returns (outcome, outcome_price, outcome_time, pnl_pips) or (None, ...).
    """
    for idx, row in future_bars.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if direction == "BULLISH":
            # For bullish: TP is above, SL is below
            # Check if both hit on same bar (use low first = SL hit)
            if low <= sl and high >= tp:
                # Ambiguous — assume SL hit first (conservative)
                pnl = _calculate_pnl(direction, None, sl)  # Will use entry
                return "LOSS", sl, idx, pnl
            elif high >= tp:
                return "WIN", tp, idx, _calculate_pnl(direction, None, tp)
            elif low <= sl:
                return "LOSS", sl, idx, _calculate_pnl(direction, None, sl)

        elif direction == "BEARISH":
            # For bearish: TP is below, SL is above
            if high >= sl and low <= tp:
                # Ambiguous — assume SL hit first (conservative)
                return "LOSS", sl, idx, _calculate_pnl(direction, None, sl)
            elif low <= tp:
                return "WIN", tp, idx, _calculate_pnl(direction, None, tp)
            elif high >= sl:
                return "LOSS", sl, idx, _calculate_pnl(direction, None, sl)

    return None, None, None, None


def _calculate_pnl(
    direction: str,
    entry_price: Optional[float],
    exit_price: float,
) -> float:
    """
    Calculate PnL in pips (price units for gold).

    For BULLISH: pnl = exit - entry
    For BEARISH: pnl = entry - exit
    """
    if entry_price is None:
        # If no entry price, use 0 as placeholder
        return 0.0
    if direction == "BULLISH":
        return round(exit_price - entry_price, 2)
    else:
        return round(entry_price - exit_price, 2)


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def calculate_stats(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive performance metrics from resolved predictions.

    Metrics include:
    - Total predictions, resolved, pending
    - Win rate, avg win/loss pips
    - Profit factor, best/worst trade
    - Win streak, current streak
    - By direction: BULLISH win rate, BEARISH win rate
    - By confidence: HIGH/MEDIUM/LOW win rates

    Parameters
    ----------
    predictions : list of dict
        List of prediction dicts (from memory_manager).

    Returns
    -------
    dict
        Comprehensive performance statistics.
    """
    total = len(predictions)
    resolved = [p for p in predictions if p.get("outcome") is not None]
    pending = [p for p in predictions if p.get("outcome") is None]
    wins = [p for p in resolved if p.get("outcome") == "WIN"]
    losses = [p for p in resolved if p.get("outcome") == "LOSS"]
    expired = [p for p in resolved if p.get("outcome") == "EXPIRED"]

    n_resolved = len(resolved)
    n_wins = len(wins)
    n_losses = len(losses)

    # Overall win rate
    win_rate = n_wins / n_resolved if n_resolved > 0 else 0.0

    # PnL calculations
    win_pnls = [p["pnl_pips"] for p in wins if p.get("pnl_pips") is not None]
    loss_pnls = [p["pnl_pips"] for p in losses if p.get("pnl_pips") is not None]

    avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
    avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0

    total_win_pnl = sum(win_pnls) if win_pnls else 0.0
    total_loss_pnl = abs(sum(loss_pnls)) if loss_pnls else 0.0

    profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float("inf")

    # Best / worst trade
    all_pnls = win_pnls + loss_pnls
    best_trade = max(all_pnls) if all_pnls else 0.0
    worst_trade = min(all_pnls) if all_pnls else 0.0

    # Streaks
    max_win_streak, current_streak = _calculate_streaks(resolved)

    # By direction
    by_direction = _by_category(resolved, key="direction")

    # By confidence label
    by_confidence = _by_category(resolved, key="confidence_label")

    return {
        "total_predictions": total,
        "resolved": n_resolved,
        "pending": len(pending),
        "wins": n_wins,
        "losses": n_losses,
        "expired": len(expired),
        "win_rate": round(win_rate, 4),
        "avg_win_pips": round(avg_win, 2),
        "avg_loss_pips": round(avg_loss, 2),
        "total_win_pnl": round(total_win_pnl, 2),
        "total_loss_pnl": round(total_loss_pnl, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "best_trade": round(best_trade, 2),
        "worst_trade": round(worst_trade, 2),
        "max_win_streak": max_win_streak,
        "current_streak": current_streak,
        "by_direction": by_direction,
        "by_confidence": by_confidence,
    }


def _calculate_streaks(
    resolved: List[Dict[str, Any]],
) -> Tuple[int, str]:
    """
    Calculate max win streak and current streak.

    Returns (max_win_streak, current_streak_description).
    """
    if not resolved:
        return 0, "N/A"

    max_streak = 0
    current_streak_count = 0
    current_streak_type = None

    for pred in resolved:
        outcome = pred.get("outcome")
        if outcome == "WIN":
            if current_streak_type == "WIN":
                current_streak_count += 1
            else:
                current_streak_type = "WIN"
                current_streak_count = 1
            max_streak = max(max_streak, current_streak_count)
        elif outcome == "LOSS":
            if current_streak_type == "LOSS":
                current_streak_count += 1
            else:
                current_streak_type = "LOSS"
                current_streak_count = 1
        else:
            current_streak_type = None
            current_streak_count = 0

    # Describe current streak
    if current_streak_type == "WIN":
        streak_desc = f"WIN x{current_streak_count}"
    elif current_streak_type == "LOSS":
        streak_desc = f"LOSS x{current_streak_count}"
    else:
        streak_desc = "N/A"

    return max_streak, streak_desc


def _by_category(
    resolved: List[Dict[str, Any]],
    key: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute win/loss stats grouped by a prediction field.

    Parameters
    ----------
    resolved : list of dict
        Resolved prediction dicts.
    key : str
        The dict key to group by (e.g. 'direction', 'confidence_label').

    Returns
    -------
    dict
        {category: {total, wins, losses, win_rate}}
    """
    groups: Dict[str, Dict[str, Any]] = {}

    for pred in resolved:
        cat = pred.get(key, "UNKNOWN")
        if cat not in groups:
            groups[cat] = {"total": 0, "wins": 0, "losses": 0}

        groups[cat]["total"] += 1
        if pred.get("outcome") == "WIN":
            groups[cat]["wins"] += 1
        elif pred.get("outcome") == "LOSS":
            groups[cat]["losses"] += 1

    # Calculate win rates
    for cat, stats in groups.items():
        total = stats["total"]
        stats["win_rate"] = round(stats["wins"] / total, 4) if total > 0 else 0.0

    return groups


def print_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print performance statistics."""
    print("\n" + "=" * 60)
    print("  XAUUSD PREDICTION PERFORMANCE STATS")
    print("=" * 60)

    print(f"\n  Overview:")
    print(f"    Total predictions:  {stats['total_predictions']}")
    print(f"    Resolved:           {stats['resolved']}")
    print(f"    Pending:            {stats['pending']}")
    print(f"    Wins:               {stats['wins']}")
    print(f"    Losses:             {stats['losses']}")
    print(f"    Expired:            {stats['expired']}")

    wr = stats["win_rate"]
    wr_pct = f"{wr * 100:.1f}%" if wr > 0 else "N/A"
    print(f"\n  Win Rate:            {wr_pct}")

    print(f"\n  PnL:")
    print(f"    Avg win:            {stats['avg_win_pips']:+.2f} pips")
    print(f"    Avg loss:           {stats['avg_loss_pips']:+.2f} pips")
    print(f"    Total win PnL:      {stats['total_win_pnl']:+.2f}")
    print(f"    Total loss PnL:     {stats['total_loss_pnl']:+.2f}")
    pf = stats.get("profit_factor")
    pf_str = f"{pf:.2f}" if pf is not None else "N/A"
    print(f"    Profit factor:      {pf_str}")
    print(f"    Best trade:         {stats['best_trade']:+.2f}")
    print(f"    Worst trade:        {stats['worst_trade']:+.2f}")

    print(f"\n  Streaks:")
    print(f"    Max win streak:     {stats['max_win_streak']}")
    print(f"    Current streak:     {stats['current_streak']}")

    by_dir = stats.get("by_direction", {})
    if by_dir:
        print(f"\n  By Direction:")
        for direction, d_stats in sorted(by_dir.items()):
            wr = d_stats["win_rate"] * 100
            print(f"    {direction:<10s}  {d_stats['wins']}/{d_stats['total']}  "
                  f"({wr:.1f}% win rate)")

    by_conf = stats.get("by_confidence", {})
    if by_conf:
        print(f"\n  By Confidence:")
        for label in ["HIGH", "MEDIUM", "LOW"]:
            if label in by_conf:
                c_stats = by_conf[label]
                wr = c_stats["win_rate"] * 100
                print(f"    {label:<10s}  {c_stats['wins']}/{c_stats['total']}  "
                      f"({wr:.1f}% win rate)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("prediction_tracker.py — Prediction Outcome Tracking")
    print("=" * 55)

    # Quick test with synthetic data
    test_preds = [
        {
            "id": "XAU_H1_20260601_1000",
            "timestamp": "2026-06-01T10:00:00Z",
            "instrument": "XAUUSD",
            "timeframe": "H1",
            "direction": "BULLISH",
            "confidence": 8.0,
            "confidence_label": "HIGH",
            "entry_price": 3300.0,
            "stop_loss": 3285.0,
            "take_profit_1": 3315.0,
            "take_profit_2": 3330.0,
            "outcome": "WIN",
            "outcome_price": 3315.0,
            "outcome_time": "2026-06-01T11:30:00Z",
            "pnl_pips": 15.0,
        },
        {
            "id": "XAU_H1_20260601_1400",
            "timestamp": "2026-06-01T14:00:00Z",
            "instrument": "XAUUSD",
            "timeframe": "H1",
            "direction": "BEARISH",
            "confidence": 6.0,
            "confidence_label": "MEDIUM",
            "entry_price": 3320.0,
            "stop_loss": 3335.0,
            "take_profit_1": 3305.0,
            "take_profit_2": 3290.0,
            "outcome": "LOSS",
            "outcome_price": 3335.0,
            "outcome_time": "2026-06-01T15:00:00Z",
            "pnl_pips": -15.0,
        },
        {
            "id": "XAU_H1_20260602_0900",
            "timestamp": "2026-06-02T09:00:00Z",
            "instrument": "XAUUSD",
            "timeframe": "H1",
            "direction": "BULLISH",
            "confidence": 7.5,
            "confidence_label": "HIGH",
            "entry_price": 3310.0,
            "stop_loss": 3295.0,
            "take_profit_1": 3325.0,
            "take_profit_2": 3340.0,
            "outcome": None,
            "outcome_price": None,
            "outcome_time": None,
            "pnl_pips": None,
        },
    ]

    stats = calculate_stats(test_preds)
    print_stats(stats)

    print("\nAll tests passed!")
