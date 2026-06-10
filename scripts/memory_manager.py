#!/usr/bin/env python3
"""
memory_manager.py — Persistent Prediction Storage for XAUUSD Forecast
======================================================================

Manages the predictions.json file that stores all generated predictions
with their outcomes for performance tracking and analysis.

Functions
---------
load_predictions(path)
    Load predictions from JSON file.

save_predictions(predictions, path)
    Save predictions to JSON with backup.

add_prediction(predictions, prediction_dict)
    Add a new prediction with auto-generated ID.

validate_prediction(pred)
    Ensure required fields exist in a prediction dict.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


# Required fields for a valid prediction dict
REQUIRED_FIELDS = {
    "id",
    "timestamp",
    "instrument",
    "timeframe",
    "direction",
    "confidence",
    "confidence_label",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
}

# Optional fields (populated later when outcome is known)
OPTIONAL_FIELDS = {
    "indicators",
    "confluences",
    "risks",
    "ml_direction",
    "ml_confidence",
    "session",
    "outcome",
    "outcome_price",
    "outcome_time",
    "pnl_pips",
}


def load_predictions(path: str) -> List[Dict[str, Any]]:
    """
    Load predictions from a JSON file.

    Parameters
    ----------
    path : str
        Path to predictions.json.

    Returns
    -------
    list of dict
        List of prediction dicts. Returns empty list if file doesn't exist
        or is empty.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except (json.JSONDecodeError, IOError):
        # Corrupted or empty file — return empty list
        return []


def save_predictions(predictions: List[Dict[str, Any]], path: str) -> None:
    """
    Save predictions to JSON with backup.

    Creates a backup copy (predictions_backup.json) before overwriting
    the main file to protect against data loss.

    Parameters
    ----------
    predictions : list of dict
        List of prediction dicts.
    path : str
        Path to predictions.json.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create backup if file already exists
    if os.path.exists(path):
        backup_path = path.replace(".json", "_backup.json")
        try:
            shutil.copy2(path, backup_path)
        except IOError:
            pass  # Non-critical: backup failure shouldn't stop save

    # Write predictions
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)


def generate_prediction_id(instrument: str, timeframe: str, timestamp: str) -> str:
    """
    Generate a unique prediction ID from instrument, timeframe, and timestamp.

    Format: {INSTRUMENT}_{TIMEFRAME}_{YYYYMMDD}_{HHMM}
    Example: XAU_H1_20260608_1400
    """
    # Parse the timestamp to extract date and time components
    try:
        if isinstance(timestamp, datetime):
            ts = timestamp
        else:
            # Handle ISO format with Z suffix
            ts_str = str(timestamp).replace("Z", "+00:00")
            ts = datetime.fromisoformat(ts_str)

        date_part = ts.strftime("%Y%m%d")
        time_part = ts.strftime("%H%M")
    except (ValueError, TypeError):
        # Fallback: use current time
        now = datetime.utcnow()
        date_part = now.strftime("%Y%m%d")
        time_part = now.strftime("%H%M")

    # Shorten instrument name (XAUUSD -> XAU)
    inst_short = instrument.replace("USD", "").replace("usd", "")

    return f"{inst_short}_{timeframe}_{date_part}_{time_part}"


def add_prediction(
    predictions: List[Dict[str, Any]],
    prediction_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Add a new prediction with auto-generated ID.

    If the prediction_dict doesn't have an 'id' field, one will be
    generated from instrument, timeframe, and timestamp.

    Parameters
    ----------
    predictions : list of dict
        Existing list of predictions.
    prediction_dict : dict
        New prediction to add.

    Returns
    -------
    list of dict
        Updated list of predictions (with the new one appended).

    Raises
    ------
    ValueError
        If the prediction dict is missing required fields.
    """
    # Validate the prediction
    validate_prediction(prediction_dict)

    # Auto-generate ID if not provided
    if "id" not in prediction_dict or not prediction_dict["id"]:
        prediction_dict["id"] = generate_prediction_id(
            instrument=prediction_dict.get("instrument", "XAUUSD"),
            timeframe=prediction_dict.get("timeframe", "H1"),
            timestamp=prediction_dict.get("timestamp", ""),
        )

    # Ensure outcome fields are initialized
    prediction_dict.setdefault("outcome", None)
    prediction_dict.setdefault("outcome_price", None)
    prediction_dict.setdefault("outcome_time", None)
    prediction_dict.setdefault("pnl_pips", None)

    # Ensure lists are initialized
    prediction_dict.setdefault("confluences", [])
    prediction_dict.setdefault("risks", [])
    prediction_dict.setdefault("indicators", {})

    predictions.append(prediction_dict)
    return predictions


def validate_prediction(pred: Dict[str, Any]) -> bool:
    """
    Ensure required fields exist in a prediction dict.

    Parameters
    ----------
    pred : dict
        Prediction dictionary to validate.

    Returns
    -------
    bool
        True if valid.

    Raises
    ------
    ValueError
        If any required field is missing or has an invalid value.
    """
    missing = REQUIRED_FIELDS - set(pred.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    # Validate direction
    if pred["direction"] not in ("BULLISH", "BEARISH"):
        raise ValueError(
            f"Invalid direction: {pred['direction']!r}. Must be BULLISH or BEARISH."
        )

    # Validate confidence range
    conf = pred["confidence"]
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 10):
        raise ValueError(
            f"Invalid confidence: {conf!r}. Must be a number between 0 and 10."
        )

    # Validate confidence_label
    label = pred.get("confidence_label", "")
    if label and label not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(
            f"Invalid confidence_label: {label!r}. Must be LOW, MEDIUM, or HIGH."
        )

    # Validate timeframe
    tf = pred["timeframe"]
    if tf not in ("M15", "H1", "H4", "D1"):
        raise ValueError(
            f"Invalid timeframe: {tf!r}. Must be M15, H1, H4, or D1."
        )

    # Validate numeric price fields
    for field in ("entry_price", "stop_loss", "take_profit_1", "take_profit_2"):
        val = pred.get(field)
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(
                f"Invalid {field}: {val!r}. Must be a number."
            )

    return True


def get_pending_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return predictions that haven't been resolved yet."""
    return [p for p in predictions if p.get("outcome") is None]


def get_resolved_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return predictions that have been resolved."""
    return [p for p in predictions if p.get("outcome") is not None]


def find_prediction_by_id(
    predictions: List[Dict[str, Any]], pred_id: str
) -> Optional[Dict[str, Any]]:
    """Find a prediction by its ID."""
    for p in predictions:
        if p.get("id") == pred_id:
            return p
    return None


# ---------------------------------------------------------------------------
# Main entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("memory_manager.py — Persistent Prediction Storage")
    print("=" * 55)

    # Quick test
    test_pred = {
        "id": "XAU_H1_20260608_1400",
        "timestamp": "2026-06-08T14:00:00Z",
        "instrument": "XAUUSD",
        "timeframe": "H1",
        "direction": "BULLISH",
        "confidence": 7.5,
        "confidence_label": "HIGH",
        "entry_price": 3350.50,
        "stop_loss": 3335.00,
        "take_profit_1": 3370.00,
        "take_profit_2": 3390.00,
    }

    # Validate
    is_valid = validate_prediction(test_pred)
    print(f"  validate_prediction(test_pred): {is_valid}")

    # Add to list
    preds = []
    preds = add_prediction(preds, test_pred)
    print(f"  add_prediction: {len(preds)} prediction(s)")
    print(f"  Generated ID: {preds[0]['id']}")

    # Test ID generation
    gen_id = generate_prediction_id("XAUUSD", "H1", "2026-06-08T14:00:00Z")
    print(f"  generate_prediction_id: {gen_id}")

    print("\nAll tests passed!")
