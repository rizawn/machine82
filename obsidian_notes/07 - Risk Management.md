# Risk Management

← Back to [[00 - MOC (Map of Content)]]

## RiskManager (`risk/risk_manager.py`)

### Position Sizing
- **ATR-based sizing:** Position size adjusted by Average True Range
- **Kelly criterion:** Uses **half-Kelly** (conservative) to avoid over-betting
- **Risk per trade:** 2% (from `trading_engine.py`)

### Circuit Breakers
| Rule | Threshold |
|------|-----------|
| Daily loss limit | 3% |
| Max consecutive losses | 5 |
| Max drawdown | 20% |

### Performance Metrics Computed
| Metric | Description |
|--------|-------------|
| Total Return | Overall portfolio return |
| Annualized Return | Year-normalized return |
| Max Drawdown | Largest peak-to-trough decline |
| Sharpe Ratio | Risk-adjusted return (total volatility) |
| Sortino Ratio | Risk-adjusted return (downside volatility only) |
| Calmar Ratio | Return / Max Drawdown |
| Volatility | Annualized standard deviation |
| Tail Ratio | 95th percentile / 5th percentile |
| Win Rate | Percentage of winning trades |
| Profit Factor | Gross profit / Gross loss |

## Backtest Risk Parameters
| Parameter | Value |
|-----------|-------|
| Stop-loss | 1% |
| Take-profit | 1.5% |
| Initial capital | $10,000 (simple) / $100,000 (MLRL01) |
| Risk per trade | 2% |
| Max drawdown | 20% |

## Empirical Risk Results
| Model | Return | Sharpe | MDD |
|-------|--------|--------|-----|
| Decision Tree | +10.80% | 0.604 | — |
| Logistic Regression | +16.03% | 0.486 | — |
| PPO (RL) | -13.94% | -0.183 | — |
| Buy & Hold | +126.95% | 1.485 | — |
