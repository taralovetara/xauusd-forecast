# XAUUSD Forecast — ML-Powered Gold Price Direction Prediction

ML-powered Gold (XAU/USD) price direction prediction using a **3-Concept Feature Framework** (Trend / Momentum / Cycle) with **Stacked Ensemble** (XGBoost + LightGBM + CatBoost → Logistic Regression meta-learner).

## Architecture

```
┌─────────────────────────────────────────────────┐
│              XAUUSD OHLCV Data                   │
│         (1m / 5m / 15m / H1 / H4 / D1)          │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│         Feature Engineering (~178 features)       │
│  ┌──────────┐ ┌───────────┐ ┌────────────────┐ │
│  │  Trend   │ │ Momentum  │ │     Cycle      │ │
│  │  EMA/SMA │ │ RSI/MACD  │ │ FFT/Fourier    │ │
│  │  ADX/CCI  │ │ Stochastic│ │ Seasonal      │ │
│  └──────────┘ └───────────┘ └────────────────┘ │
└───────────────────┬─────────────────────────────┘
                    ▼ MI Selection (Top 80)
┌─────────────────────────────────────────────────┐
│            Stacked Ensemble                       │
│  ┌──────────┐ ┌───────────┐ ┌────────────────┐ │
│  │ XGBoost  │ │ LightGBM  │ │   CatBoost     │ │
│  └────┬─────┘ └────┬──────┘ └───────┬────────┘ │
│       └─────────────┼────────────────┘          │
│                     ▼                             │
│          Logistic Regression                      │
│            (Meta-Learner)                        │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│     Final Prediction (BEARISH / NEUTRAL / BULL)  │
│       ML 40% + Technical Analysis 60%            │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
xauusd-forecast/
├── scripts/
│   ├── train_model_v2.py          # Model training (Stacked Ensemble)
│   ├── feature_engineering_v2.py  # 3-Concept feature engineering
│   ├── generate_features.py      # Batch feature generation
│   ├── generate_charts.py         # Analysis chart generation
│   ├── technical_analysis.py      # Technical indicators + scoring
│   ├── twelvedata_fetcher.py      # Real-time data via TwelveData API
│   ├── oanda_data.py              # OANDA broker data feed
│   ├── prediction_tracker.py      # Prediction journal & accuracy
│   ├── memory_manager.py          # Memory/state management
│   ├── aggregate_data.py          # Data aggregation utilities
│   └── live/                      # Real-time prediction pipeline
│       ├── h1_pipeline.py         # H1 timeframe live pipeline
│       ├── quick_signal.py        # Quick signal generation
│       └── trade_signal.py        # Trade signal formatting
├── models/
│   ├── xgb_v2_model.json          # XGBoost base model
│   ├── lgbm_v2_model.txt          # LightGBM base model
│   ├── cb_v2_model.cbm            # CatBoost base model
│   ├── meta_learner_v2.pkl        # Logistic Regression meta-learner
│   ├── scaler_v2.pkl              # Feature scaler
│   ├── feature_list_v2.json       # Selected feature names
│   └── training_results_v2.json  # Training metrics & results
├── data/
│   ├── XAUUSD_1m.csv              # 1-minute candles
│   ├── XAUUSD_M15.csv             # 15-minute candles
│   ├── XAUUSD_H1.csv              # 1-hour candles
│   ├── XAUUSD_H4.csv              # 4-hour candles
│   └── XAUUSD_D1.csv              # Daily candles
├── analysis/
│   └── charts/                    # Training analysis charts
├── config/
│   ├── twelvedata_config.json     # TwelveData API config
│   └── oanda_config.json          # OANDA API config
├── legacy/                        # Legacy scripts (initial simple 5m model)
│   ├── scripts/                   # Original simple ML scripts
│   ├── models/                    # Original simple models
│   └── data/                      # Original 5m merged data
├── .env                           # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 53.13% (5m), 93.96% at high confidence |
| AUC | 0.547 (5m) |
| High Confidence (≥0.70) Accuracy | 93.96% |
| High Confidence Coverage | ~2% of predictions |

> **Note**: The 5-minute model performance is limited by the 60-minute prediction window and older data (2004-2012). Adding recent data and adjusting the prediction horizon is expected to significantly improve results.

## Feature Framework (3 Concepts)

### 1. Trend
- EMA crossovers (5/10/20/50/100/200)
- ADX, DI+/DI-
- Price vs MA percentages
- Moving average slopes

### 2. Momentum
- RSI (7, 14, 21)
- MACD (line, signal, histogram)
- Stochastic K/D
- Williams %R
- CCI
- ROC and Momentum indicators

### 3. Cycle
- FFT spectral features
- Session timing (Asian/London/NY/Overlap)
- Day-of-week, month patterns
- Rolling statistics (std, skew)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your TwelveData API key

# Generate features
python scripts/generate_features.py

# Train model
python scripts/train_model_v2.py

# Run live prediction
python scripts/live/quick_signal.py
```

## API Keys Required

| Service | Purpose | Config File |
|---------|---------|-------------|
| TwelveData | Real-time gold price data | `.env` → `TWELVEDATA_API_KEY` |
| OANDA | Broker data feed (optional) | `config/oanda_config.json` |
