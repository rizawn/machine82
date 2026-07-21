# Feature Engineering

← Back to [[00 - MOC (Map of Content)]]

## Production Pipeline (`features/features.py`)
- **Total Features:** 47+
- **Anti-Leakage Rule:** All features use `shift(1)` lag (previous bar values only)
- **Function:** `build_production_features()`

## Feature Categories

### Trend Features
- Close vs EMA20/EMA50 (ATR-normalized)
- EMA slopes
- Trend strength
- ADX
- Trend persistence (up/down streaks)

### Volatility Features
- Rolling volatility (20d/60d annualized)
- Volatility ratio
- Volatility percentile
- ATR(14)

### Mean Reversion
- Bollinger Band z-score
- VWAP distance

### Momentum
- RSI(14) normalized
- MACD normalized
- MACD histogram normalized

### Higher-Order Statistics
- Rolling skewness (window=20)
- Rolling kurtosis (window=20)

### Market Structure
- Distance from rolling high/low
- Higher-High / Lower-Low streaks
- Breakout volume

### Volume Features
- Volume z-score
- Volume trend

### Multi-Timeframe
- Weekly return
- Weekly trend
- Daily-weekly alignment

### Signal Quality
- Tradeable trend (ADX > 20)
- Tradeable volatility (percentile 20-85)
- Combined quality score

### Regime Flags
- Trending (binary)
- Volatile (binary)
- Sideways (binary)

### Advanced Features
- Rolling Sharpe ratio (window=20)
- Rolling drawdown (window=60)
- Lag-1 autocorrelation
- Hurst exponent approximation (R/S method)
- Volatility clustering (GARCH proxy)

## Indicator Functions (`features/indicators.py`)
Reusable standalone functions:
- Returns & log returns
- Range calculations
- Moving averages
- Volatility (rolling)
- RSI (multi-period)
- ATR
- MACD
- Bollinger Bands
- SMA crossover signals

## Regime Detection (`features/regime_detection.py`)
- **Trend regime:** Price vs MA comparison
- **Volatility regime:** Rolling vol quantile
- **Combined regime:** 4 states (sideways, steady_up, crash, volatile_rally)
- **Statistical regime:** Quantile-based with smoothing
- **Wrapper:** `add_regime_features()`

## Simple Feature Set (`trading_engine.py` / `financial_ml_ohlcv.ipynb`)
8 basic features used in simpler pipeline:
1. Returns
2. Range (high-low)
3. MA10
4. MA50
5. Volatility
6. MA crossover signal
7. Volume trend
8. Price deviation from mean
