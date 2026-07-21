# Monte Carlo Simulation

← Back to [[00 - MOC (Map of Content)]]

## Simulator (`monte_carlo/simulator.py`)
- **Class:** `MonteCarloSimulator`
- **Purpose:** Stress-test strategies by simulating thousands of alternative equity paths

## Simulation Methods (6 total)

### 1. Block Bootstrap
- **Mechanism:** Samples blocks of consecutive returns (preserves serial correlation)
- **Default block size:** 20 bars
- **Why:** Simple random resampling destroys time-dependency; blocks preserve it

### 2. Regime-Aware Bootstrap
- **Mechanism:** Splits returns by volatility regime, samples within each regime
- **Why:** Different market regimes have different return distributions

### 3. Trade Resample
- **Mechanism:** Resamples individual trade PnLs
- **Use case:** Tests if strategy edge holds under different trade sequences

### 4. Return Perturbation
- **Mechanism:** Adds Gaussian noise to daily returns
- **Use case:** Tests sensitivity to small variations in returns

### 5. Cost Variation
- **Mechanism:** Randomizes transaction costs per trade
- **Use case:** Tests robustness to changing market conditions (spread widening, etc.)

### 6. Stress Test
- **Mechanism:** Injects extreme events:
  - **Flash crash:** -5% move at 1% probability
  - **Spread explosion:** 5x normal spread
- **Use case:** Worst-case scenario analysis

## Report Metrics
| Metric | Description |
|--------|-------------|
| Mean return | Average across all simulations |
| Median return | Middle value (less sensitive to outliers) |
| P(positive) | Probability of ending positive |
| P(ruin > 10%) | Probability of >10% loss |
| P(ruin > 20%) | Probability of >20% loss |
| Mean max drawdown | Average worst drawdown |
| Worst max drawdown | Worst-case drawdown |
| Sharpe statistics | Mean, std, percentiles of Sharpe ratios |

## Output Plots
- **Equity fan chart:** 5th-95th percentile bands over time
- **Return histogram:** Distribution of final returns
- **Drawdown histogram:** Distribution of max drawdowns
- **Sharpe histogram:** Distribution of Sharpe ratios

## Empirical Results (Block Bootstrap)
| Metric | Value |
|--------|-------|
| Mean return | -13.22% |
| P(positive) | 19.1% |
| Worst max drawdown | -61.77% |
