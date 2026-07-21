# Reinforcement Learning (PPO)

← Back to [[00 - MOC (Map of Content)]]

## PPO Agent (`agents/ppo_agent.py`)
- **Wrapper:** `PPOAgent` around stable-baselines3
- **Policies supported:**
  - `MlpPolicy` — standard PPO
  - `MlpLstmPolicy` — RecurrentPPO (LSTM) via sb3-contrib (graceful fallback if not installed)
- **Hyperparameters:**
  - Learning rate: 3e-4
  - n_steps: 2048
  - batch_size: 64
  - gamma (discount): 0.99
  - GAE lambda: 0.95
  - entropy coefficient: 0.01

## RecurrentPPO (LSTM)
- **File:** `train_wf.py`
- **Algorithm:** RecurrentPPO with LSTM
- **Network:** `[128, 64]` for both policy and value
- **LSTM hidden size:** 128
- **Training:** 100,000 timesteps per fold
- **TODO:** Optimize sequence lengths (12-24 bars recommended)

## Trading Environment (`env/trading_env.py`)

### Action Space (5-action discrete)
| Action | Meaning |
|--------|---------|
| 0 | Flat (no position) |
| 1 | Long 50% |
| 2 | Long 100% |
| 3 | Short 50% |
| 4 | Short 100% |

### Observation Space
- Feature vector (N dimensions from feature engineering)
- 4 portfolio state dimensions:
  1. Position percentage
  2. Unrealized PnL
  3. Drawdown
  4. Steps since entry / 20

### Reward Function: Differential Sharpe Ratio (DSR)
- **EMA** of returns and squared returns for online Sharpe estimation
- **Penalties:**
  - Transaction cost penalty (lambda_cost = 5.0)
  - Drawdown penalty (quadratic after 2% threshold, lambda_dd = 2.0)
  - Overtrading penalty (if trade rate > 15% of bars, lambda_overtrade = 0.5)

### Constraints
- **min_hold_period:** 5 bars
- **cooldown_after_loss:** 3 bars
- **max_drawdown_kill:** 20% (agent stops if portfolio drops 20%)

### Execution Realism (Friction)
- Fee: 0.01%
- Spread: 0.03%
- Slippage: 0.02%
- **Total friction per round-trip:** 0.06% (applied on both entry and exit)

### Anti-Leakage
- Observations use `current_step - 1` (previous completed bar)

## RL Training Results
- **PPO performance:** -13.94% return, -0.183 Sharpe
- **Underperformance causes:**
  - Insufficient timesteps (50K-100K may not be enough)
  - Transaction costs eat into returns
  - Gold bull market favored passive holding over active trading
- **Walk-Forward (13 folds):** Average Sharpe 0.026 ± 0.468 (stable but near-zero)

## Simple RL Proof-of-Concept (`tesrl.py`)
- Custom Gymnasium `TradingEnv`
- 3 actions (Hold/Buy/Sell)
- 5-bar close price window as observation
- Trained 10,000 timesteps
- Saved as `ppo_trading_gold.zip`
- **Note:** Reward always 0 — scaffold/proof-of-concept only

## TODO / Roadmap
1. Scale RL training to 500K-1M timesteps
2. Implement RecurrentPPO natively with optimized sequence lengths (12-24 bars)
3. Tighten trade rate penalty to reduce overtrading
