# Earnings Consolidated Final Dataset - Data Dictionary

**File**: `earnings_consolidated_final.csv`  
**Created**: 2025-11-22  
**Size**: 671.9 MB  
**Format**: CSV (UTF-8)

## Overview

| Metric | Value |
|--------|-------|
| **Total Rows** | 622,125 |
| **Total Columns** | 104 |
| **Tickers** | 31 |
| **Earnings Events** | 369 |
| **Time Range** | 2022-10-25 to 2025-11-05 (3+ years) |
| **Avg Events/Ticker** | 11.9 |
| **Avg Rows/Event** | 1,686 (±30 min window) |

## Tickers Included

**31 Quality Tickers** (HIGH + MEDIUM data quality):

```
AAPL, ADBE, AMD, AVGO, BAC, C, COST, CRM, FDX, GOOG, INTC, JPM, MDB, META, MS, 
MSFT, MU, NFLX, NKE, NOW, NVDA, ORCL, QCOM, SBUX, SHOP, SNOW, SOFI, TGT, UPST, 
WFC, WMT
```

## Column Categories

### 1. OHLCV Data (7 columns)
Core price and volume data at 1-second resolution.

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `timestamp` | datetime64 | Bar timestamp (UTC) | 0.0% |
| `open` | float64 | Open price | 0.0% |
| `high` | float64 | High price | 0.0% |
| `low` | float64 | Low price | 0.0% |
| `close` | float64 | Close price | 0.0% |
| `vol` | int64 | Volume | 0.0% |
| `vw` | float64 | Volume-weighted average price | 0.0% |
| `num` | int64 | Number of trades | 0.0% |

### 2. Metadata (4 columns)

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `ticker` | object | Stock ticker symbol | 0.0% |
| `acceptance_datetime_utc` | datetime64 | Earnings announcement time (UTC) | 0.0% |
| `EPS Estimate` | float64 | Analyst EPS estimate | 1.2% |
| `Reported EPS` | float64 | Actual reported EPS | 0.5% |
| `Surprise(%)` | float64 | EPS surprise percentage | 0.5% |

### 3. Time Windows (5 columns)
Boolean flags indicating different time periods.

| Column | Type | Description |
|--------|------|-------------|
| `5_days_before` | bool | 5 trading days before earnings |
| `5_days_after` | bool | 5 trading days after earnings |
| `event_plus_minus_30m` | bool | ±30 min around earnings |
| `next_open_plus_minus_30m` | bool | ±30 min around next market open |
| `event_plus_30_to_120m` | bool | 30-120 min after earnings |
| `seconds_since_event` | float64 | Seconds since earnings announcement |

### 4. Event-Specific Signals (11 columns)
Features calculated per earnings event.

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `event_price` | float64 | Price at earnings announcement | 0.0% |
| `or_high` | float64 | Opening range high (first 5 min) | 0.9% |
| `or_low` | float64 | Opening range low (first 5 min) | 0.9% |
| `or_width_pct` | float64 | Opening range width as % of event price | 0.9% |
| `impulse_high` | float64 | Highest high in impulse bar | 0.0% |
| `impulse_low` | float64 | Lowest low in impulse bar | 0.0% |
| `impulse_range_pct` | float64 | Impulse bar range as % | 0.0% |
| `first_5m_high` | float64 | Highest price in first 5 min | 0.7% |
| `first_5m_low` | float64 | Lowest price in first 5 min | 0.7% |
| `first_5m_range_pct` | float64 | First 5 min range as % | 0.7% |
| `cvd_since_event` | float64 | Cumulative volume delta since earnings | 51.7% |
| `cvd_pct_volume` | float64 | CVD as % of cumulative volume | 51.7% |
| `vwap_since_event` | float64 | VWAP since earnings | 51.7% |

### 5. Technical Indicators - Normalized (16 columns)
Cross-ticker comparable indicators (% or relative values).

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `rsi` | float64 | Relative Strength Index (14 period) | 0.8% |
| `roc` | float64 | Rate of Change | 0.4% |
| `atr_pct` | float64 | ATR as % of price | 0.4% |
| `vol_ratio` | float64 | Current vol / rolling avg vol | 0.0% |
| `ema_short_above_mid` | int64 | 1 if short EMA > mid EMA | 0.0% |
| `ema_mid_above_long` | int64 | 1 if mid EMA > long EMA | 0.0% |
| `ema_bullish_alignment` | int64 | 1 if all EMAs aligned bullish | 0.0% |
| `price_above_ema_short_pct` | float64 | Distance from short EMA (%) | 0.0% |
| `price_above_ema_mid_pct` | float64 | Distance from mid EMA (%) | 0.0% |
| `price_above_ema_long_pct` | float64 | Distance from long EMA (%) | 0.0% |
| `vwap_distance_pct` | float64 | Distance from VWAP (%) | 51.7% |

### 6. Technical Indicators - Raw Values (5 columns)
Original indicator values (ticker-specific, not normalized).

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `atr` | float64 | Average True Range (raw) | 0.4% |
| `ema_short` | float64 | Short EMA (raw price) | 0.0% |
| `ema_mid` | float64 | Mid EMA (raw price) | 0.0% |
| `ema_long` | float64 | Long EMA (raw price) | 0.0% |
| `vol_ma` | float64 | Volume moving average (raw) | 0.0% |

### 7. CVD & Delta Features (5 columns)
Cumulative volume delta and momentum indicators.

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `cvd_zscore` | float64 | CVD z-score within event | 51.5% |
| `delta` | float64 | Change in CVD per bar | 0.0% |
| `delta_20s` | float64 | Rolling 20-bar delta sum | 0.0% |
| `delta_10s` | float64 | Rolling 10-bar delta sum | 0.0% |

### 8. Baseline Statistics (22 columns)
Pre-earning period statistics for adaptive thresholds.

**Volatility**:
- `baseline_atr_pct_mean`, `baseline_atr_pct_std`
- `baseline_range_pct_mean`, `baseline_range_pct_std`, `baseline_range_pct_p95`

**Volume**:
- `baseline_vol_mean`, `baseline_vol_std`
- `baseline_vol_p25`, `baseline_vol_p50`, `baseline_vol_p75`

**RSI**:
- `baseline_rsi_mean`, `baseline_rsi_std`
- `baseline_rsi_p05`, `baseline_rsi_p95`

**Moves**:
- `baseline_5m_move_pct_mean`, `baseline_5m_move_pct_std`, `baseline_5m_move_pct_p95`
- `baseline_30m_move_pct_mean`, `baseline_30m_move_pct_std`, `baseline_30m_move_pct_p95`

**VWAP**:
- `baseline_vwap_dist_pct_mean`, `baseline_vwap_dist_pct_std`

All baseline stats: **0.0% NaN**

### 9. Targets (7 columns)
Forward-looking returns and hit targets.

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `target_ret_30s` | float64 | 30-second forward return | 0.0% |
| `target_ret_60s` | float64 | 60-second forward return | 0.7% |
| `target_ret_120s` | float64 | 2-minute forward return | 0.9% |
| `target_ret_240s` | float64 | 4-minute forward return | 1.2% |
| `target_ret_600s` | float64 | 10-minute forward return | 1.9% |
| `target_1R_1pct` | int64 | Hit 1% profit within 30 min | 0.0% |
| `target_2R_1pct` | int64 | Hit 2% profit within 30 min | 0.0% |

### 10. Signals - Fixed Thresholds (6 columns)
Original strategy signals with fixed parameters.

| Column | Type | Description |
|--------|------|-------------|
| `signal_eir_long` | int64 | EIR long signal (fixed vol_mult=3.0) |
| `signal_eir_short` | int64 | EIR short signal |
| `signal_impulse_long` | int64 | Impulse long signal (fixed range=0.2%) |
| `signal_impulse_short` | int64 | Impulse short signal |
| `signal_fade_long` | int64 | Fade long signal (fixed thresholds) |
| `signal_fade_short` | int64 | Fade short signal |
| `signal_cvd_long` | int64 | CVD momentum long (z > 1.2) |
| `signal_cvd_short` | int64 | CVD momentum short (z < -1.2) |

All signals: **0.0% NaN**

### 11. Signals - Adaptive Thresholds (6 columns)
Strategy signals using baseline stats for ticker-specific thresholds.

| Column | Type | Description |
|--------|------|-------------|
| `signal_eir_adaptive_long` | int64 | EIR long (volume z-score) |
| `signal_eir_adaptive_short` | int64 | EIR short |
| `signal_impulse_adaptive_long` | int64 | Impulse long (mean + 1.5×std range) |
| `signal_impulse_adaptive_short` | int64 | Impulse short |
| `signal_fade_adaptive_long` | int64 | Fade long (adaptive all params) |
| `signal_fade_adaptive_short` | int64 | Fade short |

All signals: **0.0% NaN**

### 12. Auxiliary (4 columns)

| Column | Type | Description |
|--------|------|-------------|
| `trading_session` | object | Regular/PreMarket/PostMarket |
| `earnings_events` | int64 | Number of events for this ticker |
| `pre_earning_rows` | int64 | Rows used for baseline calculation |
| `peak_1m_vol_5m` | float64 | Peak 1-min volume in first 5 min |
| `vwap_uptime` | float64 | Time above VWAP |
| `mae_1800s` | float64 | Max adverse excursion (30 min) |
| `mfe_1800s` | float64 | Max favorable excursion (30 min) |
| `hit_1pct_30m` | int64 | Hit 1% within 30 min |
| `hit_2pct_30m` | int64 | Hit 2% within 30 min |

## Usage Notes

### Filtering Data

```python
# Event windows only (±30 min around earnings)
event_data = df[df['event_plus_minus_30m'] == True]

# Pre-earning baseline periods
baseline_data = df[df['5_days_before'] == True]

# Specific ticker
aapl_data = df[df['ticker'] == 'AAPL']
```

### Working with Signals

```python
# Get all adaptive EIR long entries
eir_trades = df[df['signal_eir_adaptive_long'] == 1]

# Compare fixed vs adaptive
fixed_count = df['signal_eir_long'].sum()
adaptive_count = df['signal_eir_adaptive_long'].sum()
```

### High NaN Columns

Columns with >50% NaN are **expected** due to:
- `cvd_since_event`, `cvd_zscore`, `vwap_since_event`: Early bars in each earning event don't have enough data
- These are calculated **per event** starting from earnings time

## Data Quality

- ✅ **Complete ticker coverage**: All 31 tickers present
- ✅ **Balanced events**: 4-14 events per ticker (avg 11.9)
- ✅ **3+ years data**: Oct 2022 - Nov 2025
- ✅ **High-resolution**: 1-second bars
- ✅ **All signals available**: Both fixed and adaptive thresholds

## File Location

Production: `/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_final.csv`
