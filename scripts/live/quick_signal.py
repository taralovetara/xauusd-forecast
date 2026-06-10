#!/usr/bin/env python3
"""
quick_signal.py — Quick One-Line XAUUSD Signal Generator
==========================================================

Generates a compact, single-line signal for quick reference.

Example output:
  XAUUSD H1: ▲ BULLISH @ $3,350 | SL: $3,335 | TP: $3,370 | Conf: 7.5/10 (HIGH)
"""

from __future__ import annotations

import os
import sys

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(SCRIPTS_DIR)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, BASE_DIR)


def generate_quick_signal(prediction: dict) -> str:
    """
    Generate a one-line signal string from a prediction dict.

    Parameters
    ----------
    prediction : dict
        A prediction dict (as stored in memory/predictions.json).

    Returns
    -------
    str
        One-line signal string.
    """
    direction = prediction.get("direction", "N/A")
    arrow = "▲" if direction == "BULLISH" else "▼" if direction == "BEARISH" else "◆"
    entry = prediction.get("entry_price", 0)
    sl = prediction.get("stop_loss", 0)
    tp1 = prediction.get("take_profit_1", 0)
    confidence = prediction.get("confidence", 0)
    label = prediction.get("confidence_label", "N/A")
    timeframe = prediction.get("timeframe", "H1")

    # Format entry/SL/TP as compact prices
    if entry >= 1000:
        entry_str = f"${entry:,.0f}"
        sl_str = f"${sl:,.0f}"
        tp_str = f"${tp1:,.0f}"
    else:
        entry_str = f"${entry:.2f}"
        sl_str = f"${sl:.2f}"
        tp_str = f"${tp1:.2f}"

    signal = (
        f"XAUUSD {timeframe}: {arrow} {direction} @ {entry_str} "
        f"| SL: {sl_str} | TP: {tp_str} "
        f"| Conf: {confidence:.1f}/10 ({label})"
    )

    return signal


def main():
    """Run the H1 pipeline and output only the quick signal line."""
    # Import and run the main pipeline
    from h1_pipeline import main as run_pipeline

    # Run the full pipeline (it prints the detailed output)
    prediction = run_pipeline()

    # Generate and print the quick signal
    signal = generate_quick_signal(prediction)
    print("\n" + "=" * 70)
    print("QUICK SIGNAL:")
    print(signal)
    print("=" * 70)

    return signal


if __name__ == "__main__":
    main()
