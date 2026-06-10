#!/usr/bin/env python3
"""
trade_signal.py — Clear Trade Signal Generator
================================================

Generates a clear, actionable trade signal every time.

Output format:
  ┌─────────────────────────────────────────────┐
  │  XAUUSD TRADE SIGNAL                        │
  │  ▲ BUY @ $4,300 | SL: $4,279 | TP: $4,321 │
  │  Confidence: 7.5/10 (HIGH)                  │
  │  Action: ENTER LONG                         │
  └─────────────────────────────────────────────┘
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


def get_trade_action(direction: str, confidence: float, ml_confidence: float, agreement: bool) -> str:
    """
    Determine the trade action based on direction, confidence, and ML agreement.

    Returns one of:
    - "ENTER LONG" — Strong bullish, high confidence, tech+ML agree
    - "ENTER SHORT" — Strong bearish, high confidence, tech+ML agree
    - "BIAS LONG" — Bullish but low confidence or disagreement
    - "BIAS SHORT" — Bearish but low confidence or disagreement
    - "STAY OUT" — No clear edge, conflicting signals, very low confidence
    """
    if confidence >= 7.0 and agreement:
        if direction == "BULLISH":
            return "ENTER LONG"
        else:
            return "ENTER SHORT"
    elif confidence >= 4.0:
        if direction == "BULLISH":
            return "BIAS LONG"
        else:
            return "BIAS SHORT"
    elif confidence >= 2.0:
        if direction == "BULLISH":
            return "LEAN LONG (weak)"
        else:
            return "LEAN SHORT (weak)"
    else:
        return "STAY OUT — no edge"


def generate_trade_signal(prediction: dict, ml_direction: str = "", ml_confidence: float = 0.0, agreement: bool = False) -> str:
    """
    Generate a clear trade signal from a prediction dict.

    Parameters
    ----------
    prediction : dict
        A prediction dict from the pipeline.
    ml_direction : str
        ML model direction ("BULLISH" or "BEARISH").
    ml_confidence : float
        ML confidence (0-1).
    agreement : bool
        Whether technical and ML agree.

    Returns
    -------
    str
        Formatted trade signal string.
    """
    direction = prediction.get("direction", "N/A")
    confidence = prediction.get("confidence", 0)
    label = prediction.get("confidence_label", "N/A")
    entry = prediction.get("entry_price", 0)
    sl = prediction.get("stop_loss", 0)
    tp1 = prediction.get("take_profit_1", 0)
    tp2 = prediction.get("take_profit_2", 0)
    rr = prediction.get("rr_ratio", "2.0")

    # Determine trade action
    action = get_trade_action(direction, confidence, ml_confidence, agreement)

    # Direction formatting
    if direction == "BULLISH":
        arrow = "▲"
        trade_word = "BUY"
    else:
        arrow = "▼"
        trade_word = "SELL"

    # Confidence bar
    filled = int(confidence)
    empty = 10 - filled
    conf_bar = "█" * filled + "░" * empty

    # Risk level
    if confidence >= 7.0:
        risk = "LOW RISK"
    elif confidence >= 4.0:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK — caution"

    # Format output
    output = f"""
╔═══════════════════════════════════════════════════════╗
║            XAUUSD TRADE SIGNAL                        ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║   {arrow} {trade_word} @ ${entry:,.2f}                          ║
║                                                       ║
║   Stop Loss:  ${sl:,.2f}                              ║
║   Take Profit 1: ${tp1:,.2f}                          ║
║   Take Profit 2: ${tp2:,.2f}                          ║
║   Risk:Reward  1:{rr}                              ║
║                                                       ║
║   Confidence: {conf_bar} {confidence:.1f}/10 ({label})    ║
║   Risk Level: {risk}                     ║
║                                                       ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║   >>> ACTION: {action:<38}║
║                                                       ║
╚═══════════════════════════════════════════════════════╝"""

    return output


def generate_compact_signal(prediction: dict, ml_direction: str = "", ml_confidence: float = 0.0, agreement: bool = False) -> str:
    """Generate a compact one-line signal."""
    direction = prediction.get("direction", "N/A")
    confidence = prediction.get("confidence", 0)
    label = prediction.get("confidence_label", "N/A")
    entry = prediction.get("entry_price", 0)
    sl = prediction.get("stop_loss", 0)
    tp1 = prediction.get("take_profit_1", 0)

    action = get_trade_action(direction, confidence, ml_confidence, agreement)
    arrow = "▲" if direction == "BULLISH" else "▼"

    return f"{arrow} {direction} @ ${entry:,.2f} | SL ${sl:,.2f} | TP ${tp1:,.2f} | Conf {confidence:.1f}/10 ({label}) | {action}"


if __name__ == "__main__":
    from h1_pipeline import main as run_pipeline
    prediction = run_pipeline()
    signal = generate_trade_signal(prediction)
    print(signal)
