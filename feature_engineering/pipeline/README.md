# Feature Engineering Pipeline

This directory contains the production data processing pipeline for earnings-based trading signal generation.

## Pipeline Scripts (Run in Order)

### 1. Process Quality Tickers
**Script**: `1_process_quality_tickers.py`

Processes 31 quality tickers through the complete feature engineering pipeline:
- Loads merged OHLCV data
- Filters to earnings event windows (±30 min)
- Calculates event-specific features (OR, impulse, CVD, VWAP)
- Adds normalized features (cross-ticker comparable)
- Calculates continuous indicators (RSI, ROC, ATR, EMAs)
- Creates target variables (forward returns)

**Output**: `earnings_quality_tickers_final.csv`

---

### 2. Calculate Baseline Statistics
**Script**: `2_calculate_baseline_stats.py`

Calculates ticker-level baseline statistics from pre-earning periods:
- Uses data from 5 days before until 30 min before each earnings event
- Computes mean, std, percentiles for volatility, volume, RSI, moves
- Creates ticker-specific reference values for adaptive thresholds

**Output**: `data/baseline_stats/ticker_baseline_stats.csv`

---

### 3. Merge Baseline Statistics
**Script**: `3_merge_baseline_stats.py`

Merges baseline statistics into the earnings dataset:
- Inner join on ticker
- Each row gets its ticker's baseline stats (broadcast)
- Enables adaptive threshold calculations

**Output**: `earnings_quality_tickers_with_baseline.csv`

---

### 4. Generate Adaptive Signals
**Script**: `4_generate_adaptive_signals.py`

Generates trading signals using adaptive thresholds:
- EIR (Opening Range Breakout) - volume z-score
- Impulse Bar Breakout - mean + 1.5×std range
- Overreaction Fade - all thresholds adaptive

Uses baseline stats for ticker-specific parameters.

**Output**: `earnings_with_adaptive_signals.csv`

---

### 5. Consolidate Datasets
**Script**: `5_consolidate_datasets.py`

Creates final consolidated dataset by merging useful columns from all intermediate files:
- Adds CVD/delta features
- Adds raw technical indicators
- Adds fixed threshold signals (for comparison)
- Results in complete feature set

**Output**: `earnings_consolidated_final.csv`

---

## Quick Start

```bash
# Run entire pipeline
cd feature_engineering/pipeline
python 1_process_quality_tickers.py
python 2_calculate_baseline_stats.py
python 3_merge_baseline_stats.py
python 4_generate_adaptive_signals.py
python 5_consolidate_datasets.py
```

## Dependencies

All scripts require:
- `pandas`, `numpy`
- Custom modules: `feature_engineering.continuous_features`, `strategies.signal_generators_adaptive`

## Data Flow

```
Original Merged OHLCV
    ↓ (1_process)
earnings_quality_tickers_final.csv (56 cols)
    ↓ (2_calculate + 3_merge)
earnings_quality_tickers_with_baseline.csv (80 cols)
    ↓ (4_generate)
earnings_with_adaptive_signals.csv (87 cols)
    ↓ (5_consolidate)
earnings_consolidated_final.csv (104 cols) ← FINAL
```

## Output Location

All outputs saved to: `/Volumes/Extreme SSD/trading_data/stock/data/processed/`
