# Walk-Forward Validation

← Back to [[00 - MOC (Map of Content)]]

## Concept
Walk-forward validation (also called rolling cross-validation) is a time-series specific validation method that:
1. Trains on a sliding window of historical data
2. Tests on the subsequent out-of-sample period
3. Repeats across the entire history

## Implementation (`agents/train.py`)

### Configuration
| Parameter | Value |
|-----------|-------|
| Train window | 3 years |
| Test window | 1 year |
| Embargo gap | 60 bars |
| Method | Purged sliding window |

### Process
```
[Train: Year 1-3] → [Test: Year 4]
         slide →
    [Train: Year 2-4] → [Test: Year 5]
              slide →
         [Train: Year 3-5] → [Test: Year 6]
```

### For Each Fold
1. Split data with embargo gap
2. Train all 7 ML classifiers
3. Train PPO RL agent
4. Backtest on test period
5. Record metrics (Sharpe, Return, etc.)

## RecurrentPPO Walk-Forward (`train_wf.py`)
- **Algorithm:** RecurrentPPO (LSTM) per fold
- **Training:** 100,000 timesteps per fold
- **LSTM config:** hidden=128, net_arch=[128,64]

## Results
| Metric | Value |
|--------|-------|
| Total folds | 13 |
| Average Sharpe | 0.026 |
| Sharpe std | 0.468 |
| Interpretation | Stable but near-zero edge |

## Output
- `walk_forward_results.csv` — per-fold metrics
- Per-fold prediction charts
- Fold comparison summary

## Why Walk-Forward Matters
- **Random CV** assumes i.i.d. data — financial data is NOT i.i.d.
- **Single split** can be lucky/unlucky — walk-forward shows consistency
- **Near-zero average Sharpe** across folds = strategy has no persistent edge
- **High variance** (std=0.468) = performance varies wildly by period
