# Data Audit Report
*Generated: 2025-11-22 16:46:38*

**Dataset**: `earnings_with_adaptive_signals.csv`

---

# Data Completeness
## Tickers: 31
```
AAPL, ADBE, AMD, AVGO, BAC, C, COST, CRM, FDX, GOOG, INTC, JPM, MDB, META, MS, MSFT, MU, NFLX, NKE, NOW, NVDA, ORCL, QCOM, SBUX, SHOP, SNOW, SOFI, TGT, UPST, WFC, WMT
```
## Time Range
- Start: 2022-10-25 19:31:23+00:00
- End: 2025-11-05 21:31:11+00:00
- Duration: 1107 days

## Earnings Events: 369
- Mean events/ticker: 11.9
- Min events/ticker: 4
- Max events/ticker: 14

## Rows per Ticker
- Total rows: 622,125
- Mean rows/ticker: 20,069
- Min rows/ticker: 1,676
- Max rows/ticker: 39,632

### Per-Ticker Summary

| Ticker | Events | Rows | Avg Rows/Event |
|--------|--------|------|----------------|
| AAPL | 13 | 32,011 | 2462 |
| ADBE | 12 | 20,051 | 1671 |
| AMD | 11 | 32,263 | 2933 |
| AVGO | 12 | 16,866 | 1406 |
| BAC | 13 | 3,366 | 259 |
| C | 13 | 28,267 | 2174 |
| COST | 12 | 5,434 | 453 |
| CRM | 7 | 15,263 | 2180 |
| FDX | 13 | 18,575 | 1429 |
| GOOG | 14 | 36,396 | 2600 |
| INTC | 13 | 39,632 | 3049 |
| JPM | 12 | 2,747 | 229 |
| MDB | 13 | 16,128 | 1241 |
| META | 13 | 37,624 | 2894 |
| MS | 12 | 2,464 | 205 |
| MSFT | 10 | 28,156 | 2816 |
| MU | 12 | 34,940 | 2912 |
| NFLX | 12 | 25,617 | 2135 |
| NKE | 12 | 26,794 | 2233 |
| NOW | 13 | 8,366 | 644 |
| NVDA | 12 | 36,561 | 3047 |
| ORCL | 10 | 25,839 | 2584 |
| QCOM | 13 | 27,516 | 2117 |
| SBUX | 14 | 22,455 | 1604 |
| SHOP | 4 | 1,972 | 493 |
| SNOW | 12 | 27,613 | 2301 |
| SOFI | 13 | 12,474 | 960 |
| TGT | 12 | 6,318 | 526 |
| UPST | 13 | 22,345 | 1719 |
| WFC | 12 | 1,676 | 140 |
| WMT | 12 | 6,396 | 533 |

# Feature Inventory
**Total columns**: 87

## Price OHLCV (27 columns)

- `baseline_vol_mean` (float64) - 0.0% NaN
- `baseline_vol_p25` (float64) - 0.0% NaN
- `baseline_vol_p50` (float64) - 0.0% NaN
- `baseline_vol_p75` (float64) - 0.0% NaN
- `baseline_vol_std` (float64) - 0.0% NaN
- `baseline_vwap_dist_pct_mean` (float64) - 0.0% NaN
- `baseline_vwap_dist_pct_std` (float64) - 0.0% NaN
- `close` (float64) - 0.0% NaN
- `cvd_pct_volume` (float64) - 51.7% NaN
- `first_5m_high` (float64) - 0.7% NaN
- `first_5m_low` (float64) - 0.7% NaN
- `high` (float64) - 0.0% NaN
- `impulse_high` (float64) - 0.0% NaN
- `impulse_low` (float64) - 0.0% NaN
- `low` (float64) - 0.0% NaN
- `next_open_plus_minus_30m` (bool) - 0.0% NaN
- `num` (int64) - 0.0% NaN
- `open` (float64) - 0.0% NaN
- `or_high` (float64) - 0.9% NaN
- `or_low` (float64) - 0.9% NaN
- `peak_1m_vol_5m` (float64) - 0.7% NaN
- `vol` (int64) - 0.0% NaN
- `vol_ratio` (float64) - 0.0% NaN
- `vw` (float64) - 0.0% NaN
- `vwap_distance_pct` (float64) - 51.7% NaN
- `vwap_since_event` (float64) - 51.7% NaN
- `vwap_uptime` (float64) - 0.0% NaN

## Time (3 columns)

- `acceptance_datetime_utc` (datetime64[ns, UTC]) - 0.0% NaN
- `seconds_since_event` (float64) - 0.0% NaN
- `timestamp` (datetime64[ns, UTC]) - 0.0% NaN

## Metadata (4 columns)

- `EPS Estimate` (float64) - 1.2% NaN
- `Reported EPS` (float64) - 0.5% NaN
- `Surprise(%)` (float64) - 0.5% NaN
- `ticker` (object) - 0.0% NaN

## Event Windows (4 columns)

- `5_days_after` (bool) - 0.0% NaN
- `5_days_before` (bool) - 0.0% NaN
- `event_plus_minus_30m` (bool) - 0.0% NaN
- `next_open_plus_minus_30m` (bool) - 0.0% NaN

## Event Signals (11 columns)

- `cvd_since_event` (float64) - 51.7% NaN
- `event_price` (float64) - 0.0% NaN
- `first_5m_high` (float64) - 0.7% NaN
- `first_5m_low` (float64) - 0.7% NaN
- `impulse_high` (float64) - 0.0% NaN
- `impulse_low` (float64) - 0.0% NaN
- `impulse_range_pct` (float64) - 0.0% NaN
- `or_high` (float64) - 0.9% NaN
- `or_low` (float64) - 0.9% NaN
- `or_width_pct` (float64) - 0.9% NaN
- `vwap_since_event` (float64) - 51.7% NaN

## Normalized Features (22 columns)

- `atr_pct` (float64) - 0.4% NaN
- `baseline_30m_move_pct_mean` (float64) - 0.0% NaN
- `baseline_30m_move_pct_p95` (float64) - 0.0% NaN
- `baseline_30m_move_pct_std` (float64) - 0.0% NaN
- `baseline_5m_move_pct_mean` (float64) - 0.0% NaN
- `baseline_5m_move_pct_p95` (float64) - 0.0% NaN
- `baseline_5m_move_pct_std` (float64) - 0.0% NaN
- `baseline_atr_pct_mean` (float64) - 0.0% NaN
- `baseline_atr_pct_std` (float64) - 0.0% NaN
- `baseline_range_pct_mean` (float64) - 0.0% NaN
- `baseline_range_pct_p95` (float64) - 0.0% NaN
- `baseline_range_pct_std` (float64) - 0.0% NaN
- `baseline_vwap_dist_pct_mean` (float64) - 0.0% NaN
- `baseline_vwap_dist_pct_std` (float64) - 0.0% NaN
- `cvd_pct_volume` (float64) - 51.7% NaN
- `first_5m_range_pct` (float64) - 0.7% NaN
- `impulse_range_pct` (float64) - 0.0% NaN
- `or_width_pct` (float64) - 0.9% NaN
- `price_above_ema_long_pct` (float64) - 0.0% NaN
- `price_above_ema_mid_pct` (float64) - 0.0% NaN
- `price_above_ema_short_pct` (float64) - 0.0% NaN
- `vwap_distance_pct` (float64) - 51.7% NaN

## Technical Indicators (16 columns)

- `atr_pct` (float64) - 0.4% NaN
- `baseline_atr_pct_mean` (float64) - 0.0% NaN
- `baseline_atr_pct_std` (float64) - 0.0% NaN
- `baseline_rsi_mean` (float64) - 0.0% NaN
- `baseline_rsi_p05` (float64) - 0.0% NaN
- `baseline_rsi_p95` (float64) - 0.0% NaN
- `baseline_rsi_std` (float64) - 0.0% NaN
- `ema_bullish_alignment` (int64) - 0.0% NaN
- `ema_mid_above_long` (int64) - 0.0% NaN
- `ema_short_above_mid` (int64) - 0.0% NaN
- `price_above_ema_long_pct` (float64) - 0.0% NaN
- `price_above_ema_mid_pct` (float64) - 0.0% NaN
- `price_above_ema_short_pct` (float64) - 0.0% NaN
- `roc` (float64) - 0.4% NaN
- `rsi` (float64) - 0.8% NaN
- `vol_ratio` (float64) - 0.0% NaN

## Baseline Stats (22 columns)

- `baseline_30m_move_pct_mean` (float64) - 0.0% NaN
- `baseline_30m_move_pct_p95` (float64) - 0.0% NaN
- `baseline_30m_move_pct_std` (float64) - 0.0% NaN
- `baseline_5m_move_pct_mean` (float64) - 0.0% NaN
- `baseline_5m_move_pct_p95` (float64) - 0.0% NaN
- `baseline_5m_move_pct_std` (float64) - 0.0% NaN
- `baseline_atr_pct_mean` (float64) - 0.0% NaN
- `baseline_atr_pct_std` (float64) - 0.0% NaN
- `baseline_range_pct_mean` (float64) - 0.0% NaN
- `baseline_range_pct_p95` (float64) - 0.0% NaN
- `baseline_range_pct_std` (float64) - 0.0% NaN
- `baseline_rsi_mean` (float64) - 0.0% NaN
- `baseline_rsi_p05` (float64) - 0.0% NaN
- `baseline_rsi_p95` (float64) - 0.0% NaN
- `baseline_rsi_std` (float64) - 0.0% NaN
- `baseline_vol_mean` (float64) - 0.0% NaN
- `baseline_vol_p25` (float64) - 0.0% NaN
- `baseline_vol_p50` (float64) - 0.0% NaN
- `baseline_vol_p75` (float64) - 0.0% NaN
- `baseline_vol_std` (float64) - 0.0% NaN
- `baseline_vwap_dist_pct_mean` (float64) - 0.0% NaN
- `baseline_vwap_dist_pct_std` (float64) - 0.0% NaN

## Targets (7 columns)

- `target_1R_1pct` (int64) - 0.0% NaN
- `target_2R_1pct` (int64) - 0.0% NaN
- `target_ret_120s` (float64) - 0.9% NaN
- `target_ret_240s` (float64) - 1.2% NaN
- `target_ret_30s` (float64) - 0.0% NaN
- `target_ret_600s` (float64) - 1.9% NaN
- `target_ret_60s` (float64) - 0.7% NaN

## Adaptive Signals (6 columns)

- `signal_eir_adaptive_long` (int64) - 0.0% NaN
- `signal_eir_adaptive_short` (int64) - 0.0% NaN
- `signal_fade_adaptive_long` (int64) - 0.0% NaN
- `signal_fade_adaptive_short` (int64) - 0.0% NaN
- `signal_impulse_adaptive_long` (int64) - 0.0% NaN
- `signal_impulse_adaptive_short` (int64) - 0.0% NaN

## Uncategorized (8 columns)

- `earnings_events` - 0.0% NaN
- `event_plus_30_to_120m` - 0.0% NaN
- `hit_1pct_30m` - 0.0% NaN
- `hit_2pct_30m` - 0.0% NaN
- `mae_1800s` - 0.3% NaN
- `mfe_1800s` - 0.3% NaN
- `pre_earning_rows` - 0.0% NaN
- `trading_session` - 0.0% NaN

# Intermediate File Comparison

## Columns in Intermediate Files Not in Final

### From `with_derived` (4 columns)

- `cvd_zscore`
- `delta`
- `delta_10s`
- `delta_20s`

### From `with_signals` (12 columns)

- `cvd_zscore`
- `delta`
- `delta_10s`
- `delta_20s`
- `signal_cvd_long`
- `signal_cvd_short`
- `signal_eir_long`
- `signal_eir_short`
- `signal_fade_long`
- `signal_fade_short`
- `signal_impulse_long`
- `signal_impulse_short`

## Column Count Evolution

| File | Columns |
|------|--------|
| original | 56 |
| with_baseline | 80 |
| with_derived | 61 |
| with_signals | 69 |
| final | 87 |
