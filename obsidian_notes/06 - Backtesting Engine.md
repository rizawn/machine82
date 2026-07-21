# Backtesting Engine

← Back to [[00 - MOC (Map of Content)]]

## Core Engine (`backtest/engine.py`)
- **Class:** `BacktestEngine`
- **Purpose:** Run ML model predictions through simulated trading

### Execution Rules
- **Stop-loss:** 1%
- **Take-profit:** 1.5%
- **Execution delay:** Entry at next bar (not instant)
- **Friction:** Applied on both entry and exit (fee + spread + slippage = 0.06% round-trip)

### Anomaly Detection
`detect_anomalous_candles()` scans for:
- Zero or negative prices
- Daily moves > 15%
- Extreme volume z-scores
- High == Low candles (no-range / illiquid)

## Benchmarks (`backtest/benchmarks.py`)
- **Class:** `BenchmarkRunner`
- **Strategies:**
  1. **Buy & Hold:** Full position from start to end
  2. **SMA Crossover:** MA10 / MA50 crossover strategy
  3. **Random:** 5% trade probability per bar
- **Note:** All benchmark strategies also apply friction costs for fair comparison

## Metrics (`backtest/metrics.py`)
- **Class:** `BacktestMetrics`
- Delegates to `RiskManager` for core calculations
- Adds capital tracking and cost reporting
- `compare_table()` builds DataFrame comparing all strategies

## Configuration (from `configs/config.yaml`)
| Parameter | Value |
|-----------|-------|
| Initial capital | $100,000 |
| Fee | 0.01% |
| Spread | 0.03% |
| Slippage | 0.02% |
| Random seed | 42 |

## Output Artifacts
- `risk_summary.csv` — Return, Max Drawdown, Sharpe per model
- `comparison_results.csv` — ML metrics + backtest metrics combined
- Equity curve charts (`backtest_equity.png`)
- Per-model prediction charts overlaid with price
