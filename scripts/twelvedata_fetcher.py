#!/usr/bin/env python3
"""
twelvedata_fetcher.py — Twelve Data Real-Time XAUUSD Data Fetcher
=================================================================

Provides near real-time XAUUSD OHLCV data from Twelve Data API.
~1-2 minute latency (vs ~10 min with yfinance).

Free tier: 800 API calls/day (~160 forecast runs/day).

Usage:
    from twelvedata_fetcher import TwelveDataFetcher

    fetcher = TwelveDataFetcher()
    df_m15 = fetcher.get_candles(granularity="15min", count=500)
    df_h1  = fetcher.get_candles(granularity="1h", count=1000)
    price  = fetcher.get_current_price()
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

CONFIG_PATH = os.path.join(BASE_DIR, "config", "twelvedata_config.json")
OANDA_CONFIG_PATH = os.path.join(BASE_DIR, "config", "oanda_config.json")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Default API key (user-provided)
DEFAULT_API_KEY = "c2f7813446f540fb9b6cd83e5072b2a6"


class TwelveDataFetcher:
    """
    Near real-time XAUUSD data fetcher using Twelve Data REST API.

    Latency: ~1-2 minutes (vs ~10 min with yfinance).
    Free tier: 800 API calls/day.

    Supports:
    - Fetching OHLCV candles at any granularity
    - Getting current spot price
    - Saving data to CSV in the project's data/ directory
    """

    # Twelve Data interval mapping
    GRANULARITY_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
        "1w": "1week",
        "1mo": "1month",
        # Direct formats
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1day",
    }

    # CSV column name mapping (for pipeline compatibility)
    CSV_TIMEFRAME_MAP = {
        "15min": "XAUUSD_M15.csv",
        "1h": "XAUUSD_H1.csv",
        "4h": "XAUUSD_H4.csv",
        "1day": "XAUUSD_D1.csv",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Twelve Data fetcher.

        Parameters
        ----------
        api_key : str, optional
            Twelve Data API key. Falls back to config file, then default.
        """
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.symbol = "XAU/USD"
        self._load_api_key()

    def _load_api_key(self):
        """Load API key from config, env var, or use default."""
        # Try config file
        if not self.api_key and os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    config = json.load(f)
                self.api_key = config.get("api_key", "")
            except Exception:
                pass

        # Try environment variable
        if not self.api_key:
            self.api_key = os.environ.get("TWELVEDATA_API_KEY", "")

        # Use default
        if not self.api_key:
            self.api_key = DEFAULT_API_KEY

    @property
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def _api_call(self, endpoint: str, params: dict) -> dict:
        """Make an API call to Twelve Data."""
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            raise ConnectionError(f"Twelve Data API error: {response.status_code}")

        data = response.json()

        # Check for API errors
        if "status" in data and data["status"] == "error":
            raise ValueError(f"Twelve Data API error: {data.get('message', 'Unknown')}")

        return data

    def get_candles(
        self,
        granularity: str = "15min",
        count: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data from Twelve Data.

        Parameters
        ----------
        granularity : str
            Candle granularity: 1min, 5min, 15min, 30min, 1h, 4h, 1day, 1week
        count : int
            Number of candles to fetch (max 5000)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: open, high, low, close, volume
            Index: datetime (UTC, timezone-naive)
        """
        interval = self.GRANULARITY_MAP.get(granularity, granularity)

        params = {
            "symbol": self.symbol,
            "interval": interval,
            "outputsize": min(count, 5000),
            "timezone": "UTC",
        }

        data = self._api_call("time_series", params)

        if "values" not in data:
            raise ValueError(f"No candle data returned: {data}")

        records = []
        for candle in data["values"]:
            ts = pd.to_datetime(candle["datetime"], utc=True).tz_localize(None)
            records.append({
                "datetime": ts,
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": int(candle.get("volume", 0) or 0),
            })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df.set_index("datetime", inplace=True)
            df.index.name = "datetime"
            # Sort ascending (oldest first) for pipeline compatibility
            df.sort_index(inplace=True)

        return df

    def get_current_price(self) -> dict:
        """
        Get current real-time price for XAU/USD.

        Returns
        -------
        dict
            {"mid": float, "timestamp": datetime, "source": "twelvedata"}
        """
        data = self._api_call("price", {"symbol": self.symbol})

        if "price" not in data:
            raise ValueError(f"No price data returned: {data}")

        price = float(data["price"])

        # Get timestamp from latest 1min candle
        ts_data = self._api_call("time_series", {
            "symbol": self.symbol,
            "interval": "1min",
            "outputsize": 1,
            "timezone": "UTC",
        })

        ts = datetime.now(timezone.utc)
        if "values" in ts_data and len(ts_data["values"]) > 0:
            ts = pd.to_datetime(ts_data["values"][0]["datetime"], utc=True).tz_localize(None)

        lag = (datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)).total_seconds()

        return {
            "mid": price,
            "timestamp": ts,
            "lag_seconds": int(lag),
            "source": "twelvedata",
        }

    def refresh_all_data(self, save_csv: bool = True) -> dict:
        """
        Fetch and save all timeframes needed by the pipeline.

        Returns
        -------
        dict
            {"M15": df, "H1": df, "H4": df, "D1": df, "current_price": dict}
        """
        print("[TwelveData] Refreshing all XAUUSD data...")

        result = {}
        api_calls = 0

        # M15 — 500 candles (~5 days)
        print("  Fetching M15 candles...")
        df_m15 = self.get_candles(granularity="15min", count=500)
        result["M15"] = df_m15
        api_calls += 1
        if save_csv and len(df_m15) > 0:
            df_m15.to_csv(os.path.join(DATA_DIR, "XAUUSD_M15.csv"))
        print(f"    M15: {len(df_m15)} rows, last={df_m15.index[-1]}")

        # H1 — 1000 candles (~41 days)
        print("  Fetching H1 candles...")
        df_h1 = self.get_candles(granularity="1h", count=1000)
        result["H1"] = df_h1
        api_calls += 1
        if save_csv and len(df_h1) > 0:
            df_h1.to_csv(os.path.join(DATA_DIR, "XAUUSD_H1.csv"))
        print(f"    H1: {len(df_h1)} rows, last={df_h1.index[-1]}")

        # H4 — 500 candles (~83 days)
        print("  Fetching H4 candles...")
        df_h4 = self.get_candles(granularity="4h", count=500)
        result["H4"] = df_h4
        api_calls += 1
        if save_csv and len(df_h4) > 0:
            df_h4.to_csv(os.path.join(DATA_DIR, "XAUUSD_H4.csv"))
        print(f"    H4: {len(df_h4)} rows, last={df_h4.index[-1]}")

        # D1 — 600 candles (~2.4 years)
        print("  Fetching D1 candles...")
        df_d1 = self.get_candles(granularity="1day", count=600)
        result["D1"] = df_d1
        api_calls += 1
        if save_csv and len(df_d1) > 0:
            df_d1.to_csv(os.path.join(DATA_DIR, "XAUUSD_D1.csv"))
        print(f"    D1: {len(df_d1)} rows, last={df_d1.index[-1]}")

        # Current price
        try:
            current = self.get_current_price()
            result["current_price"] = current
            api_calls += 2  # price + 1min candle
            print(f"  Live Price: ${current['mid']:,.2f}")
            print(f"  Lag: ~{current['lag_seconds']} seconds")
            print(f"  Timestamp: {current['timestamp']} UTC")
        except Exception as e:
            print(f"  Could not fetch current price: {e}")
            result["current_price"] = None

        print(f"[TwelveData] Data refresh complete! ({api_calls} API calls used)")

        return result


def refresh_live_data_smart() -> dict:
    """
    Refresh data using the best available source:
    1. OANDA (if configured) — sub-second latency
    2. Twelve Data — ~1-2 min latency
    3. yfinance — ~10 min latency (fallback)

    Returns
    -------
    dict
        {"source": str, "data": {...}}
    """
    # 1. Try OANDA
    if os.path.exists(OANDA_CONFIG_PATH):
        try:
            from oanda_data import OandaDataFetcher
            fetcher = OandaDataFetcher()
            if fetcher.is_available:
                data = fetcher.refresh_all_data()
                return {"source": "OANDA (real-time)", "data": data}
        except Exception as e:
            print(f"[OANDA] Error: {e}")

    # 2. Try Twelve Data
    try:
        fetcher = TwelveDataFetcher()
        if fetcher.is_available:
            data = fetcher.refresh_all_data()
            return {"source": "Twelve Data (~1-2 min lag)", "data": data}
    except Exception as e:
        print(f"[TwelveData] Error: {e}")
        print("[TwelveData] Falling back to yfinance...")

    # 3. Fall back to yfinance
    print("[yfinance] Refreshing data (fallback)...")
    from oanda_data import refresh_with_yfinance_fallback
    result = refresh_with_yfinance_fallback()
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Twelve Data XAUUSD Data Fetcher")
    parser.add_argument("--refresh", action="store_true", help="Refresh all data")
    parser.add_argument("--price", action="store_true", help="Get current price only")
    parser.add_argument("--test", action="store_true", help="Test API connection")
    args = parser.parse_args()

    fetcher = TwelveDataFetcher()

    if args.test:
        print("Testing Twelve Data connection...")
        if fetcher.is_available:
            try:
                price = fetcher.get_current_price()
                print(f"Connection: OK")
                print(f"Live XAU/USD: ${price['mid']:,.2f}")
                print(f"Lag: ~{price['lag_seconds']} seconds")
                print(f"Timestamp: {price['timestamp']} UTC")
            except Exception as e:
                print(f"Connection test failed: {e}")
        else:
            print("No API key configured")

    elif args.price:
        price = fetcher.get_current_price()
        print(f"XAU/USD: ${price['mid']:,.2f}")
        print(f"Lag: ~{price['lag_seconds']} seconds")
        print(f"Timestamp: {price['timestamp']} UTC")

    elif args.refresh:
        result = refresh_live_data_smart()
        print(f"\nData source: {result['source']}")

    else:
        parser.print_help()
