#!/usr/bin/env python3
"""
Prepare average trade size data for interactive exploration.

Process second bar data for each ticker/earnings event:
- Extract 3-day window around each earnings event
- Calculate avg_trade_size = vw * vol / num
- Add time-relative columns (minutes from event)
- Add time buckets (1-min and 5-min)
- Label trading hours (Pre/Regular/Post/Overnight)
- Save to single parquet file for efficient Streamlit loading
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_engineering.trading_hours import label_trading_hours

# Paths
DATA_DIR = Path('/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv')
OUTPUT_DIR = Path('data/avg_trade_size_exploration')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / 'all_events.csv.gz'

# Quality tickers
QUALITY_TICKERS = [
    # HIGH
    'AAPL', 'ADBE', 'AMD', 'AVGO', 'CRM', 'C', 'GOOG', 'INTC',
    'META', 'MSFT', 'MU', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'QCOM',
    'SBUX', 'SNOW', 'UPST', 'WMT',
    # MEDIUM
    'BAC', 'COST', 'FDX', 'JPM', 'MDB', 'MS', 'NOW',
    'SHOP', 'SOFI', 'TGT', 'WFC'
]

# Window size (days before and after event)
DAYS_BEFORE = 3
DAYS_AFTER = 1


def process_ticker(ticker_file):
    """
    Process a single ticker's second bar data.
    
    Args:
        ticker_file: Path to ticker's second bar CSV
        
    Returns:
        DataFrame with processed event windows, or None if error
    """
    ticker = ticker_file.stem.replace('_second_merged', '')
    
    print(f"\n{ticker}:", end=" ", flush=True)
    
    # Load data
    try:
        df = pd.read_csv(ticker_file)
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return None
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check for earnings datetime
    if 'acceptance_datetime_utc' not in df.columns:
        print("❌ No acceptance_datetime_utc column")
        return None
    
    # Get unique earnings events
    df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)
    earnings_times = df['acceptance_datetime_utc'].dropna().unique()
    
    if len(earnings_times) == 0:
        print("❌ No earnings events found")
        return None
    
    print(f"{len(earnings_times)} events", end=" ", flush=True)
    
    # Process each earnings event
    event_dfs = []
    
    for event_time in earnings_times:
        # Define window
        window_start = event_time - pd.Timedelta(days=DAYS_BEFORE)
        window_end = event_time + pd.Timedelta(days=DAYS_AFTER)
        
        # Filter to window
        event_mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
        event_df = df[event_mask].copy()
        
        if len(event_df) == 0:
            continue
        
        # Calculate avg_trade_size
        # Handle division by zero
        event_df['avg_trade_size'] = np.where(
            event_df['num'] > 0,
            event_df['vw'] * event_df['vol'] / event_df['num'],
            np.nan
        )
        
        # Add time-relative columns
        event_df['event_datetime'] = event_time
        event_df['minutes_from_event'] = (
            (event_df['timestamp'] - event_time).dt.total_seconds() / 60
        ).astype(int)
        
        # Time buckets (in minutes from event)
        event_df['time_bucket_1min'] = event_df['minutes_from_event']
        event_df['time_bucket_5min'] = (event_df['minutes_from_event'] // 5) * 5
        
        # Label trading hours
        event_df['trading_hours'] = label_trading_hours(event_df['timestamp'])
        
        # Add event metadata
        event_df['event_date'] = event_time.strftime('%Y-%m-%d %H:%M UTC')
        
        # Get earnings metadata (take first non-null values for this event)
        event_meta = df[df['acceptance_datetime_utc'] == event_time].iloc[0]
        event_df['eps_estimate'] = event_meta.get('EPS Estimate', np.nan)
        event_df['reported_eps'] = event_meta.get('Reported EPS', np.nan)
        event_df['surprise_pct'] = event_meta.get('Surprise(%)', np.nan)
        
        # Select columns to keep
        keep_cols = [
            'ticker', 'event_date', 'event_datetime', 'timestamp',
            'minutes_from_event', 'time_bucket_1min', 'time_bucket_5min',
            'trading_hours', 'avg_trade_size', 'vw', 'vol', 'num',
            'open', 'high', 'low', 'close',
            'eps_estimate', 'reported_eps', 'surprise_pct'
        ]
        
        event_df = event_df[keep_cols]
        event_dfs.append(event_df)
    
    if len(event_dfs) == 0:
        print("❌ No valid event windows")
        return None
    
    # Combine all events for this ticker
    ticker_df = pd.concat(event_dfs, ignore_index=True)
    
    print(f"✓ {len(event_dfs)} events, {len(ticker_df):,} rows")
    
    return ticker_df


def main():
    print("=" * 80)
    print("PREPARING AVERAGE TRADE SIZE DATA FOR EXPLORATION")
    print("=" * 80)
    print(f"Source: {DATA_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Window: {DAYS_BEFORE} days before to {DAYS_AFTER} day after each event")
    print("=" * 80)
    
    all_data = []
    processed_count = 0
    
    for ticker in QUALITY_TICKERS:
        # Find ticker file
        ticker_file = DATA_DIR / f"{ticker}_second_merged.csv"
        
        if not ticker_file.exists():
            print(f"\n{ticker}: ❌ File not found")
            continue
        
        # Process ticker
        ticker_data = process_ticker(ticker_file)
        
        if ticker_data is not None:
            all_data.append(ticker_data)
            processed_count += 1
    
    # Combine all data
    print("\n" + "=" * 80)
    print("COMBINING DATA")
    print("=" * 80)
    
    if len(all_data) == 0:
        print("❌ No data processed successfully")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Total rows: {len(combined_df):,}")
    print(f"Total events: {combined_df['event_date'].nunique()}")
    print(f"Tickers processed: {processed_count}/{len(QUALITY_TICKERS)}")
    print()
    
    # Summary statistics
    print("Data Summary:")
    print(f"  Avg trade size range: ${combined_df['avg_trade_size'].min():,.2f} - ${combined_df['avg_trade_size'].max():,.2f}")
    print(f"  Avg trade size median: ${combined_df['avg_trade_size'].median():,.2f}")
    print(f"  Trading hours distribution:")
    print(combined_df['trading_hours'].value_counts().to_string())
    print()
    
    # Save to CSV with gzip compression
    print("Saving to CSV (gzip)...", end=" ", flush=True)
    combined_df.to_csv(OUTPUT_FILE, index=False, compression='gzip')
    
    # Check file size
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"✓ ({file_size_mb:.1f} MB)")
    print()
    
    # Sample data preview
    print("Sample data (first 5 rows):")
    print(combined_df[['ticker', 'event_date', 'minutes_from_event', 
                       'trading_hours', 'avg_trade_size']].head())
    print()
    
    print("=" * 80)
    print("✅ DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Run the Streamlit app:")
    print(f"  streamlit run analysis/streamlit_trade_size_explorer.py")
    print()


if __name__ == '__main__':
    main()
