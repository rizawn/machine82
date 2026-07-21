# Project Architecture & Entry Points

← Back to [[00 - MOC (Map of Content)]]

## Overview
**MLRL01** is a production-grade Machine Learning + Reinforcement Learning trading engine optimized for **Gold Futures (GC=F)**.

**Design philosophy:** Strict realism over inflated metrics — 100% data-leakage-free modeling pipeline.

## Directory Structure
```
MLRL01/
├── agents/          # Training & agent logic (ML + RL)
├── backtest/        # Backtest execution & benchmarks
├── configs/         # Central configuration (config.yaml)
├── data/            # Raw & processed data (empty placeholders)
├── env/             # Gymnasium trading environment
├── features/        # Feature engineering pipeline
├── jupiter/         # Data files & experiment artifacts (copied from ../jupiter/)
├── models/          # Saved PPO/RecurrentPPO models
├── monte_carlo/     # Monte Carlo simulation engine
├── notebooks/       # Future research notebooks (empty)
├── reports/         # Audit documentation
├── results/         # Output plots & CSV reports
├── risk/            # Risk management
├── main.py          # Main orchestrator
├── app.py           # Streamlit web UI
├── train_wf.py      # Walk-Forward RecurrentPPO trainer
├── test_mc.py       # Monte Carlo unit test
├── quick_test.py    # Quick sanity check
└── narikapi.py      # Qwen API test (unrelated)
```

## Entry Points

### `main.py` — Full Pipeline Orchestrator
1. Load latest data
2. Build features
3. Train all ML models + PPO RL agent
4. Run backtests
5. Run benchmarks
6. Walk-forward validation
7. Monte Carlo simulations
8. Generate charts and reports

### `app.py` — Streamlit Web UI ("Monster Mode")
- Same pipeline as `main.py`
- Premium dark-gold theme
- Interactive config sliders
- Real-time progress display
- Tabbed result views:
  - Overview
  - Equity Curves
  - ML Predictions
  - Monte Carlo
  - Walk-Forward

### `train_wf.py` — Walk-Forward RecurrentPPO
- Slides configurable window across history
- Default: 3y train / 1y test
- Trains RecurrentPPO (LSTM) per fold

### `quick_test.py` — Sanity Check
- Loads data
- Builds features
- Tests environment
- Runs 50 random steps
- Mini Monte Carlo

### `test_mc.py` — Monte Carlo Unit Test
- Tests simulator with dummy returns

## Simple Pipeline (`jupiter/` — standalone)
| File | Purpose |
|------|---------|
| `call.py` | Download Gold data from yfinance |
| `sc1.py` | Download AAPL data from Alpha Vantage |
| `ex1.py` | Simple MA5 crossover signal |
| `tesrl.py` | PPO training proof-of-concept |
| `trading_engine.py` | Unified ML + RL + backtest (single script) |
| `schedule.py` | Daily data download scheduler |
| `financial_ml_ohlcv.ipynb` | Visual ML pipeline notebook |
| `DOL.pinescript` | TradingView Pine Script indicator |

## Data Flow
```
Raw OHLCV → Anomaly Detection → Feature Generation → Target Labeling
    → Purged Train/Test Split → ML + RL Training
    → Backtest (with friction) → Monte Carlo → Reports
```

## Dependencies
- **Core:** pandas, numpy, matplotlib, scikit-learn, xgboost, lightgbm, stable-baselines3, gymnasium, yfinance, pyyaml, schedule, requests, streamlit
- **Optional:** sb3-contrib (RecurrentPPO), hmmlearn (HMM regime detection)
