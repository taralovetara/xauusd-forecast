#!/usr/bin/env python3
"""
oanda_data.py — OANDA Real-Time Data Fetcher for XAUUSD
========================================================

Provides real-time XAUUSD OHLCV data from OANDA's streaming API.
Replaces yfinance with sub-second latency data.

Usage:
    from oanda_data import OandaDataFetcher

    fetcher = OandaDataFetcher()  # Reads config from config/oanda_config.json
    df_m15 = fetcher.get_candles("XAU_USD", granularity="M15", count=500)
    df_h1  = fetcher.get_candles("XAU_USD", granularity="H1", count=500)
    price  = fetcher.get_current_price("XAU_USD")

OANDA Instrument Format: "XAU_USD" (not "XAUUSD")
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = SCRIPT_DIR  # oanda_data.py is in scripts/
BASE_DIR = os.path.dirname(SCRIPTS_DIR)

CONFIG_PATH = os.path.join(BASE_DIR, "config", "oanda_config.json")
DATA_DIR = os.path.join(BASE_DIR, "data")


class OandaDataFetcher:
    """
    Real-time XAUUSD data fetcher using OANDA v20 REST API.

    Supports:
    - Fetching OHLCV candles at any granularity (M1, M5, M15, H1, H4, D)
    - Getting current mid/bid/ask price
    - Saving data to CSV in the project's data/ directory
    - Falling back to yfinance if OANDA is unavailable
    """

    # OANDA granularity mapping
    GRANULARITY_MAP = {
        "1m": "M1",
        "5m": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
        "1d": "D",
        "1w": "W",
        # Direct OANDA formats
        "M1": "M1",
        "M5": "M5",
        "M15": "M15",
        "M30": "M30",
        "H1": "H1",
        "H4": "H4",
        "D": "D",
        "W": "W",
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the OANDA data fetcher.

        Parameters
        ----------
        config_path : str, optional
            Path to oanda_config.json. Defaults to config/oanda_config.json.
        """
        self.config_path = config_path or CONFIG_PATH
        self.api_key = None
        self.account_id = None
        self.environment = "practice"  # "practice" for demo, "live" for real
        self.client = None
        self._load_config()

    def _load_config(self):
        """Load OANDA API credentials from config file or environment variables."""
        # Try config file first
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = json.load(f)
            self.api_key = config.get("api_key", "")
            self.account_id = config.get("account_id", "")
            self.environment = config.get("environment", "practice")

        # Override with environment variables if set
        self.api_key = os.environ.get("OANDA_API_KEY", self.api_key)
        self.account_id = os.environ.get("OANDA_ACCOUNT_ID", self.account_id)
        self.environment = os.environ.get("OANDA_ENVIRONMENT", self.environment)

        # Initialize client if we have credentials
        if self.api_key:
            self._init_client()

    def _init_client(self):
        """Initialize the OANDA v20 API client."""
        try:
            import oandapyV20
            from oandapyV20 import API

            if self.environment == "live":
                host = "api-fxtrade.oanda.com"
            else:
                host = "api-fxpractice.oanda.com"

            self.client = API(
                access_token=self.api_key,
                environment=self.environment,
            )
            print(f"[OANDA] Client initialized ({self.environment})")
        except Exception as e:
            print(f"[OANDA] Failed to initialize client: {e}")
            self.client = None

    @property
    def is_available(self) -> bool:
        """Check if OANDA client is ready for use."""
        return self.client is not None and self.api_key is not None

    def get_candles(
        self,
        instrument: str = "XAU_USD",
        granularity: str = "M15",
        count: int = 500,
        price: str = "M",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data from OANDA.

        Parameters
        ----------
        instrument : str
            OANDA instrument name (e.g., "XAU_USD")
        granularity : str
            Candle granularity: M1, M5, M15, M30, H1, H4, D, W
        count : int
            Number of candles to fetch (max 5000)
        price : str
            Price component: "M" (mid), "B" (bid), "A" (ask)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: open, high, low, close, volume
            Index: datetime (UTC, timezone-naive)
        """
        if not self.is_available:
            raise ConnectionError(
                "OANDA client not available. Set API key in config/oanda_config.json "
                "or OANDA_API_KEY environment variable."
            )

        import oandapyV20.endpoints.instruments as instruments

        # Map granularity
        oanda_gran = self.GRANULARITY_MAP.get(granularity, granularity)

        params = {
            "granularity": oanda_gran,
            "count": min(count, 5000),
            "price": price,
        }

        r = instruments.InstrumentCandles(instrument=instrument, params=params)
        resp = self.client.request(r)

        # Parse candles
        records = []
        for candle in resp.get("candles", []):
            if not candle.get("complete", True):
                continue  # Skip incomplete candles for historical data

            ts = pd.to_datetime(candle["time"]).tz_convert("UTC").tz_localize(None)

            if price == "M":
                o = float(candle["mid"]["o"])
                h = float(candle["mid"]["h"])
                l = float(candle["mid"]["l"])
                c = float(candle["mid"]["c"])
            elif price == "B":
                o = float(candle["bid"]["o"])
                h = float(candle["bid"]["h"])
                l = float(candle["bid"]["l"])
                c = float(candle["bid"]["c"])
            elif price == "A":
                o = float(candle["ask"]["o"])
                h = float(candle["ask"]["h"])
                l = float(candle["ask"]["l"])
                c = float(candle["ask"]["c"])
            else:
                o = float(candle["mid"]["o"])
                h = float(candle["mid"]["h"])
                l = float(candle["mid"]["l"])
                c = float(candle["mid"]["c"])

            vol = int(candle.get("volume", 0))

            records.append({
                "datetime": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": vol,
            })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df.set_index("datetime", inplace=True)
            df.index.name = "datetime"

        return df

    def get_current_price(self, instrument: str = "XAU_USD") -> dict:
        """
        Get current real-time price for an instrument.

        Returns
        -------
        dict
            {"bid": float, "ask": float, "mid": float, "timestamp": datetime}
        """
        if not self.is_available:
            raise ConnectionError("OANDA client not available.")

        import oandapyV20.endpoints.pricing as pricing

        params = {"instruments": instrument}
        r = pricing.PricingInfo(accountID=self.account_id, params=params)
        resp = self.client.request(r)

        if "prices" in resp and len(resp["prices"]) > 0:
            price_data = resp["prices"][0]
            bid = float(price_data["bids"][0]["price"])
            ask = float(price_data["asks"][0]["price"])
            mid = (bid + ask) / 2.0
            ts = pd.to_datetime(price_data["time"]).tz_convert("UTC").tz_localize(None)

            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": ask - bid,
                "timestamp": ts,
            }
        else:
            raise ValueError(f"No price data returned for {instrument}")

    def refresh_all_data(
        self,
        instrument: str = "XAU_USD",
        save_csv: bool = True,
    ) -> dict:
        """
        Fetch and save all timeframes needed by the pipeline.

        Returns
        -------
        dict
            {"M15": df, "H1": df, "H4": df, "D1": df, "current_price": dict}
        """
        print("[OANDA] Refreshing all XAUUSD data...")

        result = {}

        # M15 — 500 candles (~5 days)
        print("  Fetching M15 candles...")
        df_m15 = self.get_candles(instrument, granularity="M15", count=500)
        result["M15"] = df_m15
        if save_csv and len(df_m15) > 0:
            df_m15.to_csv(os.path.join(DATA_DIR, "XAUUSD_M15.csv"))
        print(f"    M15: {len(df_m15)} rows, last={df_m15.index[-1]}")

        # H1 — 1000 candles (~41 days)
        print("  Fetching H1 candles...")
        df_h1 = self.get_candles(instrument, granularity="H1", count=1000)
        result["H1"] = df_h1
        if save_csv and len(df_h1) > 0:
            df_h1.to_csv(os.path.join(DATA_DIR, "XAUUSD_H1.csv"))
        print(f"    H1: {len(df_h1)} rows, last={df_h1.index[-1]}")

        # H4 — 500 candles (~83 days)
        print("  Fetching H4 candles...")
        df_h4 = self.get_candles(instrument, granularity="H4", count=500)
        result["H4"] = df_h4
        if save_csv and len(df_h4) > 0:
            df_h4.to_csv(os.path.join(DATA_DIR, "XAUUSD_H4.csv"))
        print(f"    H4: {len(df_h4)} rows, last={df_h4.index[-1]}")

        # D1 — 600 candles (~2.4 years)
        print("  Fetching D1 candles...")
        df_d1 = self.get_candles(instrument, granularity="D", count=600)
        result["D1"] = df_d1
        if save_csv and len(df_d1) > 0:
            df_d1.to_csv(os.path.join(DATA_DIR, "XAUUSD_D1.csv"))
        print(f"    D1: {len(df_d1)} rows, last={df_d1.index[-1]}")

        # Current price
        try:
            current = self.get_current_price(instrument)
            result["current_price"] = current
            print(f"  Live Price: ${current['mid']:,.2f} "
                  f"(Bid: ${current['bid']:,.2f} / Ask: ${current['ask']:,.2f})")
            print(f"  Spread: ${current['spread']:.2f}")
            print(f"  Timestamp: {current['timestamp']} UTC")
        except Exception as e:
            print(f"  Could not fetch current price: {e}")
            result["current_price"] = None

        print("[OANDA] Data refresh complete!")
        return result


def refresh_with_yfinance_fallback() -> dict:
    """
    Refresh data using OANDA if available, otherwise fall back to yfinance.

    Returns
    -------
    dict
        {"source": "oanda"|"yfinance", "data": {...}}
    """
    fetcher = OandaDataFetcher()

    if fetcher.is_available:
        try:
            data = fetcher.refresh_all_data()
            return {"source": "oanda", "data": data}
        except Exception as e:
            print(f"[OANDA] Error: {e}")
            print("[OANDA] Falling back to yfinance...")

    # Fall back to yfinance
    print("[yfinance] Refreshing data (fallback)...")
    import yfinance as yf

    ticker = "GC=F"
    result = {}

    # M15
    df_15m = yf.download(ticker, interval="15m", period="60d", auto_adjust=True, progress=False)
    if isinstance(df_15m.columns, pd.MultiIndex):
        df_15m.columns = df_15m.columns.get_level_values(0)
    df_15m.columns = [c.lower() for c in df_15m.columns]
    df_15m.index = pd.to_datetime(df_15m.index, utc=True).tz_convert("UTC").tz_localize(None)
    df_15m = df_15m[["open", "high", "low", "close", "volume"]]
    df_15m.index.name = "datetime"
    df_15m.to_csv(os.path.join(DATA_DIR, "XAUUSD_M15.csv"))
    result["M15"] = df_15m
    print(f"  M15: {len(df_15m)} rows")

    # H1
    df_1h = yf.download(ticker, interval="1h", period="730d", auto_adjust=True, progress=False)
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    df_1h.columns = [c.lower() for c in df_1h.columns]
    df_1h.index = pd.to_datetime(df_1h.index, utc=True).tz_convert("UTC").tz_localize(None)
    df_1h = df_1h[["open", "high", "low", "close", "volume"]]
    df_1h.index.name = "datetime"
    df_1h.to_csv(os.path.join(DATA_DIR, "XAUUSD_H1.csv"))
    result["H1"] = df_1h
    print(f"  H1: {len(df_1h)} rows")

    # H4 from H1
    df_h4 = df_1h.copy().groupby(pd.Grouper(freq="4h")).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    df_h4.to_csv(os.path.join(DATA_DIR, "XAUUSD_H4.csv"))
    result["H4"] = df_h4
    print(f"  H4: {len(df_h4)} rows")

    # D1
    df_d1 = yf.download(ticker, interval="1d", period="max", auto_adjust=True, progress=False)
    if isinstance(df_d1.columns, pd.MultiIndex):
        df_d1.columns = df_d1.columns.get_level_values(0)
    df_d1.columns = [c.lower() for c in df_d1.columns]
    df_d1.index = pd.to_datetime(df_d1.index, utc=True).tz_convert("UTC").tz_localize(None)
    df_d1 = df_d1[["open", "high", "low", "close", "volume"]]
    df_d1.index.name = "datetime"
    df_d1.to_csv(os.path.join(DATA_DIR, "XAUUSD_D1.csv"))
    result["D1"] = df_d1
    print(f"  D1: {len(df_d1)} rows")

    # Current price
    current = yf.download(ticker, interval="1m", period="1d", auto_adjust=True, progress=False)
    if isinstance(current.columns, pd.MultiIndex):
        current.columns = current.columns.get_level_values(0)
    price = float(current["Close"].iloc[-1])
    ts = current.index[-1]
    result["current_price"] = {"mid": price, "timestamp": ts, "source": "yfinance (~10min delayed)"}
    print(f"  Price: ${price:,.2f} (yfinance, ~10min delayed)")

    return {"source": "yfinance", "data": result}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OANDA XAUUSD Data Fetcher")
    parser.add_argument("--refresh", action="store_true", help="Refresh all data")
    parser.add_argument("--price", action="store_true", help="Get current price only")
    parser.add_argument("--test", action="store_true", help="Test OANDA connection")
    parser.add_argument("--instrument", default="XAU_USD", help="OANDA instrument (default: XAU_USD)")
    args = parser.parse_args()

    if args.test:
        fetcher = OandaDataFetcher()
        if fetcher.is_available:
            print("OANDA connection: OK")
            try:
                price = fetcher.get_current_price(args.instrument)
                print(f"Live {args.instrument}: ${price['mid']:,.2f}")
                print(f"Spread: ${price['spread']:.2f}")
                print(f"Timestamp: {price['timestamp']} UTC")
            except Exception as e:
                print(f"Connection test failed: {e}")
        else:
            print("OANDA connection: NOT CONFIGURED")
            print(f"Set API key in {CONFIG_PATH} or OANDA_API_KEY env var")

    elif args.price:
        result = refresh_with_yfinance_fallback()
        price_info = result["data"].get("current_price", {})
        source = result["source"]
        print(f"Source: {source}")
        if price_info:
            print(f"XAUUSD: ${price_info.get('mid', 0):,.2f}")
            if "bid" in price_info:
                print(f"Bid: ${price_info['bid']:,.2f} | Ask: ${price_info['ask']:,.2f}")
                print(f"Spread: ${price_info['spread']:.2f}")
            print(f"Timestamp: {price_info.get('timestamp', 'N/A')}")

    elif args.refresh:
        result = refresh_with_yfinance_fallback()
        print(f"\nData source: {result['source']}")

    else:
        parser.print_help()
