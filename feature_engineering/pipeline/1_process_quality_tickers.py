#!/usr/bin/env python3
"""
Process high and medium quality tickers with earnings signals, then merge results.

This script:
1. Processes each ticker individually with add_earnings_signals_meta.py
2. Filters to earnings event windows only (event_plus_minus_30m)
3. Merges the filtered results into a single training dataset

This approach is memory-efficient since we only keep earnings windows (30-min periods).
"""

import pandas as pd
from pathlib import Path
import sys
from feature_engineering.add_earnings_signals_meta import add_earnings_signals

# HIGH quality tickers (20) - Safe for 1s resampling
HIGH_TICKERS = [
    'AAPL', 'ADBE', 'AMD', 'AVGO', 'CRM', 'C', 'GOOG', 'INTC', 
    'META', 'MSFT', 'MU', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'QCOM', 
    'SBUX', 'SNOW', 'UPST', 'WMT'
]

# MEDIUM quality tickers (11) - Consider adaptive approach
MEDIUM_TICKERS = [
    'BAC', 'COST', 'FDX', 'JPM', 'MDB', 'MS', 'NOW', 
    'SHOP', 'SOFI', 'TGT', 'WFC'
]

# Combined quality tickers (31 total)
QUALITY_TICKERS = HIGH_TICKERS + MEDIUM_TICKERS

# Paths
DATA_DIR = Path('/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv')
OUTPUT_DIR = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed')
TEMP_DIR = OUTPUT_DIR / 'temp'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUTPUT = OUTPUT_DIR / 'earnings_quality_tickers_features.csv'


def process_ticker(ticker):
    """
    Process a single ticker with earnings signals and return only event windows.
    
    Returns:
        DataFrame with filtered earnings windows, or None if error
    """
    # Find ticker file
    file_patterns = [
        DATA_DIR / f"{ticker}_second_merged.csv",
        DATA_DIR / f"{ticker}.csv",
        DATA_DIR / f"{ticker}_merged.csv"
    ]
    
    file_path = None
    for pattern in file_patterns:
        if pattern.exists():
            file_path = pattern
            break
    
    if file_path is None:
        print(f"  ✗ File not found")
        return None
    
    try:
        # Temporary files for this ticker
        temp_features = TEMP_DIR / f"{ticker}_features.csv"
        
        # Process with earnings signals
        print(f"  Processing {ticker}...", end=" ", flush=True)
        add_earnings_signals(
            input_file=str(file_path),
            output_file=str(temp_features),
            pre_labeled=True  # Data already has earnings metadata
        )
        
        # Load and filter to earnings windows only
        df = pd.read_csv(temp_features)
        
        # Filter to event windows (±30 min around earnings)
        if 'event_plus_minus_30m' in df.columns:
            df_filtered = df[df['event_plus_minus_30m'] == True].copy()
            pct_kept = (len(df_filtered) / len(df)) * 100
            print(f"✓ {len(df):,} → {len(df_filtered):,} rows ({pct_kept:.1f}% kept)")
        else:
            print(f"⚠️  No event_plus_minus_30m column, keeping all {len(df):,} rows")
            df_filtered = df
        
        # Clean up temp file
        temp_features.unlink()
        
        return df_filtered
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    print("=" * 80)
    print("EARNINGS QUALITY TICKERS - BATCH PROCESSING")
    print("=" * 80)
    print(f"Processing {len(QUALITY_TICKERS)} quality tickers (20 HIGH + 11 MEDIUM)")
    print(f"Output: {FINAL_OUTPUT}")
    print("=" * 80)
    print()
    
    all_results = []
    successful = []
    failed = []
    
    for i, ticker in enumerate(QUALITY_TICKERS, 1):
        quality = "HIGH" if ticker in HIGH_TICKERS else "MEDIUM"
        print(f"[{i}/{len(QUALITY_TICKERS)}] {ticker} ({quality})")
        
        df = process_ticker(ticker)
        
        if df is not None and len(df) > 0:
            all_results.append(df)
            successful.append(ticker)
        else:
            failed.append(ticker)
    
    # Combine all results
    if not all_results:
        print("\n❌ No data processed successfully!")
        sys.exit(1)
    
    print(f"\n{'=' * 80}")
    print("COMBINING RESULTS")
    print(f"{'=' * 80}")
    print(f"Merging {len(all_results)} ticker datasets...")
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by ticker and timestamp
    print("Sorting by ticker and timestamp...")
    if 'timestamp' in final_df.columns:
        final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], utc=True)
        final_df = final_df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    # Save final result
    print(f"Saving to: {FINAL_OUTPUT}")
    final_df.to_csv(FINAL_OUTPUT, index=False)
    
    # Summary
    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Successful: {len(successful)}/{len(QUALITY_TICKERS)} tickers")
    print(f"Total rows: {len(final_df):,}")
    print(f"Total columns: {len(final_df.columns)}")
    print(f"Tickers: {final_df['ticker'].nunique() if 'ticker' in final_df.columns else 'N/A'}")
    print(f"File size: {FINAL_OUTPUT.stat().st_size / (1024**2):.1f} MB")
    print(f"\nOutput: {FINAL_OUTPUT}")
    
    if failed:
        print(f"\n⚠️  Failed tickers ({len(failed)}): {', '.join(failed)}")
    
    # Show sample rows per ticker
    if 'ticker' in final_df.columns and len(successful) > 0:
        print(f"\nRows per ticker (earnings windows only):")
        ticker_counts = final_df['ticker'].value_counts().sort_index()
        for ticker, count in ticker_counts.items():
            quality = "HIGH" if ticker in HIGH_TICKERS else "MEDIUM"
            print(f"  {ticker:6s} ({quality:6s}): {count:,}")
    
    # Clean up temp directory
    try:
        import shutil
        if TEMP_DIR.exists() and not list(TEMP_DIR.glob('*')):
            TEMP_DIR.rmdir()
    except:
        pass


if __name__ == '__main__':
    main()
