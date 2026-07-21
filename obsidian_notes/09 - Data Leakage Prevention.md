# Data Leakage Prevention

← Back to [[00 - MOC (Map of Content)]]

## Background
The project underwent a major audit that identified and eliminated **14 critical data leakage sources**. This dropped ML accuracy from a **fake 95-100%** down to a **realistic 51-58%**.

## 14 Leakage Fixes (V4 Audit)

### Feature-Level Leakage
1. **All features use `shift(1)`** — calculations use only previous-bar data, never current or future bars
2. **Intermediate calculations** — all rolling computations use lagged values
3. **No look-ahead in indicators** — RSI, MACD, ATR all computed with shifted data

### Split-Level Leakage
4. **Time-series split (not random)** — data split chronologically, not shuffled
5. **60-bar embargo gap** — buffer zone between train and test prevents information bleed
6. **Purged walk-forward CV** — sliding window with embargo in each fold

### RL Environment Leakage
7. **Observation uses `current_step - 1`** — agent sees only previous completed bar
8. **Reward uses realized PnL only** — no future information in reward signal

### Target Labeling Leakage
9. **Triple Barrier with proper barriers** — ATR-based barriers don't peek ahead
10. **Forward return target** — label is based on future, but features are not

### Backtest Leakage
11. **Execution delay** — trades execute at next bar, not current bar
12. **Friction on entry AND exit** — realistic costs applied both ways
13. **No lookahead in signal generation** — predictions based on lagged features only

### Metric Leakage
14. **Out-of-sample evaluation only** — all reported metrics are on test data, not training data

## Anti-Leakage Checklist
- [ ] All features use `shift(1)`
- [ ] Train/test split is chronological
- [ ] Embargo gap between train and test (60 bars)
- [ ] RL observations use previous bar
- [ ] Backtest has execution delay
- [ ] Friction applied on entry and exit
- [ ] Metrics computed on test data only
- [ ] Walk-forward CV uses purged splits

## Key Insight
**Inflated accuracy is worse than no model at all.** A model showing 95% accuracy due to leakage will fail catastrophically in production. Honest metrics (51-58% for directional prediction) are the baseline for real evaluation.
