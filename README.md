# XAUUSD 5-Minute Forecast

ML-based XAUUSD (Gold/USD) price direction prediction using 5-minute candle data with technical indicator features.

## Overview

This project trains machine learning models to predict the next 5-minute candle direction (UP/DOWN) for XAUUSD using 89 technical indicator features engineered from OHLCV data.

## Dataset

- **Source**: Historical 5-minute XAUUSD candle data (2004-2012, 10 of 21 files processed)
- **Total candles**: 506,269 (merged from 10 files)
- **Training sample**: 100,000 most recent candles
- **Columns**: datetime, open, high, low, close, volume

## Features (89 total)

| Category | Features |
|----------|----------|
| Moving Averages | SMA/EMA (5, 10, 20, 50, 100, 200), MA crossovers, price-vs-MA % |
| Oscillators | RSI (7, 14, 21), Stochastic K/D, Williams %R |
| Trend | MACD (line, signal, histogram), ADX, DI+/DI-, CCI (14, 20) |
| Volatility | Bollinger Bands (20, 50), ATR (7, 14, 21), BB width & % |
| Momentum | ROC (3, 5, 10, 20), Momentum (3, 5, 10) |
| Candle | Body, range, upper/lower shadow, body-to-range ratio |
| Volume | SMA 10/50, volume ratio, OBV |
| Lag | Close lag (1,2,3,5), return lag (1,2,3,5) |
| Rolling Stats | Std & Skew (10, 20, 50) |
| Session | Asian/London/NY/Overlap, hour, day-of-week, month |

## Models

| Model | Val Accuracy | Test Accuracy | HC>60% Acc | HC>60% Coverage |
|-------|-------------|---------------|------------|-----------------|
| XGBoost | 52.62% | 53.38% | 54.93% | 13.4% |
| Random Forest | 52.98% | 53.36% | 60.87% | 2.1% |
| Logistic Regression | 52.78% | 53.73% | 54.93% | 1.4% |

### Key Findings

1. **Top features**: RSI-14, close-vs-SMA10%, close-vs-SMA20%, RSI-7, session timing
2. **Session timing matters**: Asian/London/NY session features rank in top 10
3. **High-confidence filtering works**: Filtering predictions by confidence >60% improves accuracy to 55-61%
4. **Next steps**: Add remaining 11 data files (2012-2026) to improve model performance with more recent market data

## Top 20 Feature Importance (XGBoost)

1. rsi_14 (0.0238)
2. close_vs_sma10_pct (0.0159)
3. close_vs_sma20_pct (0.0139)
4. rsi_7 (0.0136)
5. session_asian (0.0134)
6. session_london (0.0130)
7. return_lag_3 (0.0127)
8. sma_20 (0.0126)
9. ema_50 (0.0124)
10. session_overlap (0.0124)

## Project Structure

```
xauusd-forecast/
├── README.md
├── requirements.txt
├── data/                        # Raw & processed data
│   └── xauusd_5m_merged.csv    # Merged dataset
├── models/                      # Trained models
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.json
│   ├── feature_importance.json
│   ├── training_results.json
│   └── training_config.json
├── scripts/
│   ├── merge_data.py           # Merge raw XLSX files
│   ├── feature_engineering.py  # Generate technical indicators
│   ├── train.py                # Train ML models
│   └── predict.py              # Prediction pipeline
├── notebooks/
│   └── exploration.ipynb       # (placeholder)
└── results/
    └── training_results.json
```

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Predict from live data
```python
from scripts.predict import predict
import yfinance as yf

# Fetch latest 5-min data
data = yf.download('GC=F', period='60d', interval='5m')

# Predict next candle direction
result = predict(data, model_name='xgb')
print(result)
# {'direction': 'UP', 'confidence': 0.62, 'up_probability': 0.62, ...}
```

## Data Pipeline (for adding remaining files)

1. Place raw XLSX files in `data/raw/`
2. Run `python scripts/merge_data.py` to merge all files
3. Run `python scripts/feature_engineering.py` to compute features
4. Run `python scripts/train.py` to retrain models

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Trading involves significant risk. Past model performance does not guarantee future results.
