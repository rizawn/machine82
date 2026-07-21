# Data Acquisition Pipeline

← Back to [[00 - MOC (Map of Content)]]

## Sources

### 1. Yahoo Finance (yfinance)
- **Script:** `call.py`
- **Symbol:** Gold Futures (`GC=F`)
- **Range:** 2010-01-01 to present
- **Output:** Auto-incremented CSV files (`data01.csv`, `data02.csv`, etc.)
- **Behavior:** Scans existing `data*.csv`, finds max number, increments, saves and auto-opens in default viewer

### 2. Alpha Vantage API
- **Script:** `sc1.py`
- **Symbol:** AAPL (Apple Inc.)
- **Endpoint:** `TIME_SERIES_DAILY`
- **Note:** Free tier only; no free Indonesian stock API available
- **Output:** `data_saham.csv`

### 3. Data Files Present
| File | Source | Content |
|------|--------|---------|
| `data_saham.csv` | Alpha Vantage | AAPL OHLCV |
| `data01.csv` - `data06.csv` | yfinance | Gold Futures snapshots |
| `xauusd_data.csv` | Unknown | XAU/USD Gold Spot (unused) |

## Automation
- **Script:** `schedule.py`
- **Library:** `schedule`
- **Schedule:** Daily at 13:00
- **Purpose:** Auto-download fresh market data
- **Known Bug:** `schedule.every()...do(job)` is inside the `job()` function, so schedule only registers after first run

## Simple Technical Analysis
- **Script:** `ex1.py`
- **Method:** MA5 (5-day Moving Average) crossover
- **Logic:** If `last_price > ma5` → BUY, else → SELL
- **Note:** Pure Python with `csv` module, no ML, no external libraries
