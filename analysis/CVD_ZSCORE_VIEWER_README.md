# CVD Z-Score Viewer

Interactive Streamlit app for exploring CVD (Cumulative Volume Delta) z-score behavior across earnings events.

## Quick Start

```bash
# Install dependencies
pip install streamlit pandas numpy plotly

# Run the app
streamlit run streamlit_cvd_zscore_viewer.py
```

## Dependencies

| Package | Version |
|---------|---------|
| streamlit | ≥1.51.0 |
| pandas | ≥2.0.0 |
| numpy | ≥1.24.0 |
| plotly | ≥5.0.0 |

## Data Requirements

The app expects a CSV file with earnings event data. Configure the path in the sidebar under "📂 Data Source".

### Required Columns

| Column | Description |
|--------|-------------|
| `acceptance_datetime_utc` | Earnings event timestamp |
| `ticker` | Stock symbol |
| `year` | Year (or derived from timestamp) |
| `seconds_since_event` | Seconds since earnings event |
| `close` | Close price |
| `cvd_zscore` | CVD z-score |
| `cvd_since_event` | Cumulative volume delta |
| `target_ret_600s` | 10-min forward return |

### Optional Columns

| Column | Description |
|--------|-------------|
| `event_price` | Price at event time |
| `vwap_since_event` | VWAP since event |
| `vw` | Per-bar volume-weighted price |
| `first_5m_range_pct` | First 5-min price range % |
| `Surprise(%)` | Earnings surprise % |
| `volume` | Bar volume |
| `target_ret_120s` | 2-min forward return |
| `target_ret_240s` | 4-min forward return |

## App Tabs

### 1. 📊 CVD Z-Score Trajectories & Distributions
- Visualize CVD z-score paths over time for multiple events
- Compare distributions across tickers
- Summary statistics table

### 2. 📉 Ticker Correlations
- Compare CVD-return correlation strength across tickers
- Identify mean reversion vs momentum stocks
- Bar chart and histogram of correlations

### 3. 🔍 Event Deep Dive
- Single event analysis with dual-axis charts
- CVD z-score vs price return since event
- CVD z-score vs forward 10-min return
- VW price action vs CVD z-score (time series comparison)
- **5-30m Distribution Estimator**: Predict 5-30m percentiles from 0-5m data using:
  - Formula: P_late = μ_pred + (P_early - μ_early) × k
  - Global α, β (regression), k (vol decay) parameters
  - Comparison with actual 5-30m values

### 4. 🎯 Correlation Group Analysis
- Group events into strategy regimes:
  - Strong Mean Reversion
  - Moderate Mean Reversion
  - No Signal
  - Momentum
- Adjustable thresholds
- Feature distributions by group (strip plots)

### 5. 📈 Event Correlation Deep Dive
- Explore event-level factors predicting CVD-return correlation
- X-axis options: Quarter, Surprise %, Range %, CVD metrics, etc.
- Compare 10-min vs short-term correlations

### 6. 📐 VWAP Deviation Explorer
- Track VWAP deviation from 1m to 15m
- Trajectories by strategy group
- Scatter plots at key time points

### 7. 🔬 First 5m Signals Explorer
- CVD velocity & acceleration analysis
- Price-CVD divergence patterns
- Return autocorrelation (early vs later)
- VW price autocorrelation metrics

### 8. 🔄 0-5m vs 5-30m Comparison
Event-level comparison of early vs later CVD z-score behavior:
- **Directional Persistence**: Mean(0-5m) vs Mean(5-30m)
- **Trend Strength**: Skewness(0-5m) vs Skewness(5-30m)
- **Regime Stability**: % Positive obs(0-5m) vs % Positive obs(5-30m)
- **Volatility Decay**: StdDev(0-5m) vs StdDev(5-30m)

## Sidebar Controls

- **📂 Data Source**: Configure CSV file path
- **🔧 Filters**:
  - Year selection
  - Time range (seconds since event)
  - Quick presets (First 5 min, First 30 min, etc.)

## Key Metrics

- **CVD Z-Score**: Standardized cumulative volume delta, indicating buy/sell pressure relative to baseline
- **CVD-Return Correlation**: Negative = mean reversion signal, Positive = momentum signal
- **Strategy Groups**: Based on correlation thresholds (adjustable in Tab 4)

