# Trade Size Explorer Guide

This tool allows you to interactively explore average trade size distributions around earnings events, helping to identify market participant behavior (retail vs. institutional) and data anomalies.

## 1. Prerequisites

You must have the **merged second-bar OHLCV data** for the tickers you wish to analyze.

### Data Format
The input files should be CSVs named `{ticker}_second_merged.csv` (e.g., `AAPL_second_merged.csv`) containing the following columns:
- `timestamp`: UTC timestamp of the bar
- `acceptance_datetime_utc`: Earnings event timestamp
- `vw`: Volume-weighted average price
- `vol`: Volume
- `num`: Number of trades
- `open`, `high`, `low`, `close`: Price data
- `EPS Estimate`, `Reported EPS`, `Surprise(%)`: Earnings metadata

### 1.1 Merging Raw Data (If needed)
If you have raw data files split by date or event, you can use the provided script to merge them into the required single-file-per-ticker format.

1. **Configure Paths**: Open `data_preprocessing/merge_data.py` and update the constants at the top:
   ```python
   DATA_ROOT = "/path/to/raw/data"
   METADATA_PATH = os.path.join(DATA_ROOT, "metadata", "metadata.csv")
   OUTPUT_DIR = "/path/to/output/merged_ohlcv"
   ```

2. **Run the Script**:
   ```bash
   python data_preprocessing/merge_data.py
   ```
   This will read the metadata, combine all minute/second files for each ticker, add earnings info, and save them to your output directory as `{ticker}_second_merged.csv`.

## 2. Generating Dependency Data

The Streamlit app relies on a pre-processed dataset (`data/avg_trade_size_exploration/all_events.csv.gz`) for performance. You must generate this file first.

### Step A: Configure Data Path
The data generation script currently points to a specific directory. You may need to update this path to match your local environment.

1. Open `analysis/prepare_trade_size_data.py`
2. Locate the `DATA_DIR` constant near the top of the file:
   ```python
   DATA_DIR = Path('/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv')
   ```
3. Update this path to point to the folder containing your `{ticker}_second_merged.csv` files.

### Step B: Run the Script
Execute the preparation script from the project root:

```bash
python analysis/prepare_trade_size_data.py
```

**What this does:**
- Extracts a 3-day window around each earnings event (3 days before, 1 day after).
- Calculates `avg_trade_size` (`vw * vol / num`).
- Adds time buckets (1-min, 5-min) and trading hour labels.
- Saves the processed data to `data/avg_trade_size_exploration/all_events.csv.gz`.

## 3. Running the App

Once the data is generated, you can launch the interactive explorer:

```bash
streamlit run analysis/streamlit_trade_size_explorer.py
```

### Features
- **Ticker/Event Selection**: Choose specific earnings events to analyze.
- **Dual Charts**:
  1. **Trade Size Distribution**: Scatter plot of individual trade sizes over time.
  2. **Price Action**: Candlestick chart with **Round Lot Overlay** (colored dots indicating 100, 200, 300+ share lots).
- **Filters**:
  - **Time Window**: Zoom in/out (up to ±72 hours).
  - **Outlier Removal**: Filter out extreme trade sizes to see the core distribution.
  - **Log Scale**: Toggle log scale for the trade size axis.

## 4. Troubleshooting

**"Data file not found" error in Streamlit:**
- Ensure you ran `analysis/prepare_trade_size_data.py` successfully.
- Check that `data/avg_trade_size_exploration/all_events.csv.gz` exists.

**"No data found for [Ticker]" error:**
- Verify that the ticker is included in the `QUALITY_TICKERS` list in `analysis/prepare_trade_size_data.py`.
- Ensure the input CSV for that ticker exists in your `DATA_DIR`.

**Timezone Issues:**
- The app displays times in **UTC** (e.g., 20:30 for market close).
- If you see unexpected times (like 16:30), ensure you are using the latest version of the app which forces "Wall Time" display strings.
