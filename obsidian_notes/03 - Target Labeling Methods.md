# Target Labeling Methods

← Back to [[00 - MOC (Map of Content)]]

## Triple Barrier Method (De Prado, 2018)
- **Implementation:** `features/features.py` → `_triple_barrier_target()`
- **Default method** in MLRL01
- **Mechanism:**
  - Upper barrier: ATR-based profit target
  - Lower barrier: ATR-based stop-loss
  - Vertical barrier: Time limit (default 5 days)
- **Label:** Which barrier was hit first (1 = profit, 0 = loss, -1 = timeout)

## Threshold Method
- **Logic:** Forward return > threshold → 1, else → 0
- **Use case:** When you want to filter for meaningful moves only

## Binary Method
- **Logic:** Forward return > 0 → 1, else → 0
- **Use case:** Simple next-day direction prediction
- **Used in:** `financial_ml_ohlcv.ipynb` and `trading_engine.py`

## Anti-Leakage Considerations
- All target labels use **forward-looking** data by definition
- Train/test split must use **embargo gap** (60 bars) to prevent information bleed from test period into training features
- Features must use `shift(1)` so no future data leaks into the model
