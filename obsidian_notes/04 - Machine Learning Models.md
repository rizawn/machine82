# Machine Learning Models

← Back to [[00 - MOC (Map of Content)]]

## Implemented Classifiers (7 models)

| Model | Key Parameters | Notes |
|-------|---------------|-------|
| **Logistic Regression** | Default | Simple baseline, performed well (Sharpe 0.486, Return +16.03%) |
| **Decision Tree** | max_depth=5 | Best ML performer (Sharpe 0.604, Return +10.80%) |
| **Random Forest** | 200 estimators | Ensemble of decision trees |
| **Gradient Boosting** | 200 estimators | Sequential tree boosting |
| **XGBoost** | 200 estimators, logloss objective | Gradient boosting optimized |
| **LightGBM** | 200 estimators | Fast gradient boosting |
| **SVM** | RBF kernel | Support Vector Machine |

## Training Pipeline
1. Load OHLCV data
2. Build features (47+ in MLRL01, 8 in simple pipeline)
3. Create target labels
4. **80/20 time-series split** (NOT random shuffle)
5. **StandardScaler** normalization
6. Train all 7 classifiers
7. Evaluate: accuracy, precision, recall
8. Rank models by metrics

## Key Results (Post-Audit)
- **Realistic accuracy:** 51-58% (after fixing 14 data leakage sources)
- **Previously inflated:** 95-100% (due to data leakage)
- **Simple models outperformed complex ones:** Decision Tree and Logistic Regression beat Random Forest, XGBoost, etc.
- **Buy & Hold benchmark:** +126.95% return, Sharpe 1.485 (strong gold bull market 2010-2026)
- **No ML/RL model beat Buy & Hold** on Gold Futures in this dataset

## Ensemble Method
- **File:** `agents/ensemble.py`
- **Method:** Weighted majority voting across multiple models
- **Status:** Placeholder for future integration (P8: position sizing)

## Transformer Policy (Stub)
- **File:** `agents/transformer_policy.py`
- **Status:** Scaffold only — `TransformerFeaturesExtractor` inherits from `BaseFeaturesExtractor`
- **Contains:** Placeholder MLP, comments on implementing real transformer encoder

## Output Artifacts
- `comparison_results.csv` — ML metrics + backtest metrics
- Per-model prediction charts (`*_prediction.png`)
- Confusion matrices (`confusion_matrices.png`)
- Accuracy comparison bars (`accuracy_comparison.png`)
- Metrics heatmap (`metrics_heatmap.png`)
