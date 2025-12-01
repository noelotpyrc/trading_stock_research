# Earnings Consolidated Final Dataset - Data Dictionary

**Dataset Name**: Earnings Consolidated Final  
**Filename**: `earnings_consolidated_final.csv`  
**Full Path**: `/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_final.csv`  
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

#### CVD Calculation Methodology

**Delta** estimates buying vs selling pressure for each 1-second bar based on where the close price sits within the high-low range.

**Formula**:
```
position = (close - midpoint) / half_range

where:
  midpoint = (high + low) / 2
  half_range = (high - low) / 2

delta = volume × position
```

**Position Range**: Continuous real number from **-1.0** (maximum selling) to **+1.0** (maximum buying)

| Position | Interpretation | Close Location |
|----------|---------------|----------------|
| +1.0 | Strong buying pressure | Close = High |
| +0.5 | Moderate buying | Close between mid and high |
| 0.0 | Neutral | Close = Midpoint or High = Low |
| -0.5 | Moderate selling | Close between mid and low |
| -1.0 | Strong selling pressure | Close = Low |

**Example**:
- Bar: High=$100, Low=$96, Close=$99, Volume=10,000
- Midpoint = $98, Half Range = $2
- Position = (99 - 98) / 2 = **+0.5**
- Delta = 10,000 × 0.5 = **+5,000** (buying pressure)

**Cumulative Volume Delta (CVD)**:
```
cvd_since_event = cumsum(delta) from earnings announcement time
```

CVD accumulates delta values starting from the earnings announcement timestamp, providing a running total of net buying/selling pressure during the event window.

**CVD Z-Score**:
```
cvd_zscore = (cvd - rolling_mean) / rolling_std

where rolling statistics are calculated within each earnings event
```

Z-score normalizes CVD relative to its behavior within the same event, making extreme values comparable across different tickers and events. Values above +2 or below -2 indicate strong directional pressure.

**Edge Cases**:
- When `high == low` (no price movement), position = 0.0 to avoid division by zero
- Bars before the event have NaN for `cvd_since_event` and `cvd_zscore` (51.7% NaN) as defined in the calculation method

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

---

# Round Lot Features Extension

**Extended Dataset**: `earnings_consolidated_with_rl.csv`  
**Full Path**: `/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_with_rl.csv`  
**Created**: 2025-11-28  
**Total Columns**: 128 (106 original + 22 round lot features)

## What is a Round Lot?

A **round lot** is a second bar where:
- `num == 1`: Exactly one trade occurred in that second
- `vol % 100 == 0`: Volume is a multiple of 100 shares (100, 200, 300, ...)

### Why Round Lots Matter

| Characteristic | Implication |
|----------------|-------------|
| Single trade per second | Likely a **deliberate human order**, not HFT noise |
| Volume in multiples of 100 | Classic **retail or institutional** sizing pattern |
| Low frequency (3.3% of bars) | Represents **intentional** order flow |

Round lots may represent "informed" or "intentional" trading activity, contrasting with high-frequency algorithmic trading which typically produces odd lots and multiple trades per second.

### Round Lot Statistics

| Metric | Value |
|--------|-------|
| Total bars | 622,125 |
| Round lot bars | 20,801 |
| Round lot percentage | 3.3% |

---

## Round Lot Features (22 columns)

### 13. Round Lot - Rolling Volume Features (10 columns)

These measure round lot **activity** over rolling time windows.

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `rl_vol_30s` | float64 | Rolling sum of round lot volume (30s window) | 9.2% |
| `rl_vol_60s` | float64 | Rolling sum of round lot volume (60s window) | 3.5% |
| `rl_vol_120s` | float64 | Rolling sum of round lot volume (120s window) | 1.6% |
| `rl_vol_300s` | float64 | Rolling sum of round lot volume (5min window) | 0.8% |
| `rl_vol_600s` | float64 | Rolling sum of round lot volume (10min window) | 0.6% |
| `rl_vol_pct_30s` | float64 | Round lot volume as % of total volume (30s) | 9.2% |
| `rl_vol_pct_60s` | float64 | Round lot volume as % of total volume (60s) | 3.5% |
| `rl_vol_pct_120s` | float64 | Round lot volume as % of total volume (120s) | 1.6% |
| `rl_vol_pct_300s` | float64 | Round lot volume as % of total volume (5min) | 0.8% |
| `rl_vol_pct_600s` | float64 | Round lot volume as % of total volume (10min) | 0.6% |

#### Calculation Details

**`rl_vol_{W}s`** - Rolling Round Lot Volume
```
1. For each bar, check if it's a round lot: (num == 1) & (vol % 100 == 0)
2. If round lot: use vol; otherwise: use 0
3. Sum these values over the rolling window of W seconds

Pseudocode:
  rl_vol_flag = vol if is_round_lot else 0
  rl_vol_{W}s = sum(rl_vol_flag) over last W seconds
```

**Interpretation:**
- High values = lots of round lot trading activity
- Low values = quiet period for deliberate orders
- Spikes may indicate retail/institutional reaction to news or price levels

**`rl_vol_pct_{W}s`** - Rolling Round Lot Percentage
```
1. Calculate rl_vol_{W}s (rolling round lot volume)
2. Calculate total_vol_{W}s = sum(vol) over last W seconds (all bars)
3. Divide: rl_vol_{W}s / total_vol_{W}s * 100

Pseudocode:
  rl_vol_pct_{W}s = (rl_vol_{W}s / total_vol_{W}s) * 100
```

**Interpretation:**
- High % (e.g., >50%) = market dominated by deliberate human orders
- Low % (e.g., <10%) = market dominated by algorithmic/HFT activity
- Changes in this ratio may signal shifts in market participant mix

---

### 14. Round Lot - Event-Anchored CVD Features (2 columns)

These measure round lot **order flow direction** cumulative from the earnings event.

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `rl_cvd_since_event` | float64 | Cumulative round lot delta since event start | 0.0% |
| `rl_cvd_zscore` | float64 | Z-score of rl_cvd_since_event within event | 0.0% |

#### Calculation Details

**`rl_cvd_since_event`** - Cumulative Round Lot CVD
```
1. For each bar, calculate price direction: direction = sign(vw - vw_prev)
   - vw is volume-weighted average price for the bar
   - direction = +1 if price went up, -1 if down, 0 if unchanged
2. Calculate delta for each bar: delta = direction * vol
3. For round lot bars only, accumulate delta from event start:
   - If round lot: add delta to cumulative sum
   - If not round lot: delta contributes 0
4. Carry forward the cumulative sum to all bars

Pseudocode:
  direction = sign(vw - vw.shift(1))
  delta = direction * vol
  rl_delta = delta if is_round_lot else 0
  rl_cvd_since_event = cumsum(rl_delta) starting from event_time
```

**Interpretation:**
- Positive = round lots are net **BUYING** since the event
- Negative = round lots are net **SELLING** since the event
- Tracks whether deliberate human orders are accumulating or distributing

**`rl_cvd_zscore`** - Round Lot CVD Z-Score
```
1. For each event, collect all rl_cvd_since_event values
2. Calculate event_mean = mean(rl_cvd_since_event) for that event
3. Calculate event_std = std(rl_cvd_since_event) for that event
4. For each bar: rl_cvd_zscore = (rl_cvd_since_event - event_mean) / event_std

Pseudocode:
  event_mean = mean(rl_cvd_since_event) within event
  event_std = std(rl_cvd_since_event) within event
  rl_cvd_zscore = (rl_cvd_since_event - event_mean) / event_std
```

**Interpretation:**
- Values > 2: Unusually strong round lot **buying pressure**
- Values < -2: Unusually strong round lot **selling pressure**
- Normalizes across events with different volatility levels

---

### 15. Round Lot - Rolling CVD Ratio & Direction Features (10 columns)

These measure round lot **order flow relative to total flow** over rolling windows.

| Column | Type | Description | NaN % |
|--------|------|-------------|-------|
| `rl_cvd_ratio_30s` | float64 | Round lot CVD / Total CVD (30s window) | 9.2% |
| `rl_cvd_ratio_60s` | float64 | Round lot CVD / Total CVD (60s window) | 3.5% |
| `rl_cvd_ratio_120s` | float64 | Round lot CVD / Total CVD (120s window) | 1.6% |
| `rl_cvd_ratio_300s` | float64 | Round lot CVD / Total CVD (5min window) | 0.8% |
| `rl_cvd_ratio_600s` | float64 | Round lot CVD / Total CVD (10min window) | 0.6% |
| `rl_direction_30s` | float64 | Round lot flow direction (30s): +1/-1/0 | 9.2% |
| `rl_direction_60s` | float64 | Round lot flow direction (60s): +1/-1/0 | 3.5% |
| `rl_direction_120s` | float64 | Round lot flow direction (120s): +1/-1/0 | 1.6% |
| `rl_direction_300s` | float64 | Round lot flow direction (5min): +1/-1/0 | 0.8% |
| `rl_direction_600s` | float64 | Round lot flow direction (10min): +1/-1/0 | 0.6% |

#### Calculation Details

**`rl_cvd_ratio_{W}s`** - Rolling Round Lot CVD Ratio
```
1. Calculate delta for each bar: delta = sign(vw - vw_prev) * vol
2. Calculate rl_delta: delta if is_round_lot else 0
3. Sum rl_delta over rolling window: rl_cvd_{W}s = sum(rl_delta) over last W seconds
4. Sum delta over rolling window: total_cvd_{W}s = sum(delta) over last W seconds
5. Divide with epsilon to avoid division by zero:
   rl_cvd_ratio_{W}s = rl_cvd_{W}s / (total_cvd_{W}s + epsilon)

Pseudocode:
  delta = sign(vw - vw.shift(1)) * vol
  rl_delta = delta if is_round_lot else 0
  rl_cvd_{W}s = sum(rl_delta) over last W seconds
  total_cvd_{W}s = sum(delta) over last W seconds
  rl_cvd_ratio_{W}s = rl_cvd_{W}s / (total_cvd_{W}s + epsilon)
```

**Interpretation:**
- `> 1`: Round lots **more directional** than overall market
- `≈ 1`: Round lots aligned with market
- `< 1` (same sign): Round lots less directional than market
- **Opposite signs**: Round lots trading **AGAINST** the market (potential reversal signal)

**`rl_direction_{W}s`** - Rolling Round Lot Direction
```
1. Calculate rl_cvd_{W}s = sum(rl_delta) over last W seconds
2. Apply sign function: +1 if positive, -1 if negative, 0 if zero

Pseudocode:
  rl_direction_{W}s = sign(rl_cvd_{W}s)
```

**Interpretation:**
- `+1`: Round lots net **BUYING** in the recent window
- `-1`: Round lots net **SELLING** in the recent window
- `0`: Round lots balanced (no net direction)

---

## Usage Examples

### Analyzing Round Lot Activity

```python
# High round lot activity periods
high_rl_activity = df[df['rl_vol_pct_60s'] > 30]  # >30% of volume from round lots

# Round lots diverging from market
divergence = df[
    (df['rl_direction_60s'] == 1) &  # Round lots buying
    (df['cvd_zscore'] < -1)           # But overall market selling
]
```

### Round Lot Flow Analysis

```python
# Strong round lot buying pressure
strong_rl_buying = df[df['rl_cvd_zscore'] > 2]

# Round lots driving the market (ratio > 1)
rl_driven = df[df['rl_cvd_ratio_120s'] > 1]
```

### Combining with Targets

```python
# Correlation between round lot features and forward returns
rl_features = ['rl_vol_pct_60s', 'rl_cvd_zscore', 'rl_cvd_ratio_60s', 'rl_direction_60s']
targets = ['target_ret_10s', 'target_ret_20s', 'target_ret_30s']

correlations = df[rl_features + targets].corr()[targets].loc[rl_features]
```

---

## Feature Generation Script

**Script**: `feature_engineering/add_round_lot_features.py`

**Usage**:
```bash
python feature_engineering/add_round_lot_features.py \
  --input /path/to/earnings_consolidated_final.csv \
  --output /path/to/earnings_consolidated_with_rl.csv \
  --windows 30,60,120,300,600 \
  --min-periods 10
```

**Parameters**:
- `--windows`: Comma-separated list of rolling window sizes in seconds
- `--min-periods`: Minimum observations in window to produce a value (default: 10)
