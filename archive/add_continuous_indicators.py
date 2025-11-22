#!/usr/bin/env python3
"""
Add continuous technical indicators to the processed earnings dataset.

This script:
1. Loads the merged earnings features file (622K rows, 46 columns)
2. Adds continuous indicators (RSI, EMA, ATR, etc.) per ticker with adaptive parameters
3. Saves enhanced dataset with ~10 additional indicator columns

Quality-based parameters:
- HIGH tickers (20): 1s resampling, standard periods (RSI-14, EMA-21)
- MEDIUM tickers (11): 2s resampling, adjusted periods (RSI-10, EMA-15)
"""

import pandas as pd
from pathlib import Path
import sys
from feature_engineering.continuous_features import add_continuous_indicators, get_ticker_quality

# Paths
INPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_features.csv')
OUTPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_indicators.csv')


def main():
    print("=" * 80)
    print("ADDING CONTINUOUS INDICATORS TO EARNINGS DATASET")
    print("=" * 80)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)
    print()
    
    # Load the merged file
    print(f"Loading {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Process each ticker separately (different quality tiers need different parameters)
    print("Processing by ticker with adaptive parameters...")
    print("=" * 80)
    
    results = []
    tickers = sorted(df['ticker'].unique())
    
    for i, ticker in enumerate(tickers, 1):
        quality = get_ticker_quality(ticker)
        ticker_df = df[df['ticker'] == ticker].copy()
        
        print(f"[{i}/{len(tickers)}] {ticker} ({quality}, {len(ticker_df):,} rows)")
        
        # Add continuous indicators
        ticker_df = add_continuous_indicators(ticker_df, ticker_quality=quality)
        
        results.append(ticker_df)
    
    print()
    print("=" * 80)
    print("MERGING RESULTS")
    print("=" * 80)
    
    # Combine all tickers
    final_df = pd.concat(results, ignore_index=True)
    
    # Sort by ticker and timestamp
    print("Sorting by ticker and timestamp...")
    final_df = final_df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Total rows: {len(final_df):,}")
    print(f"Total columns: {len(final_df.columns)} (was {len(df.columns)}, added {len(final_df.columns) - len(df.columns)})")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    print(f"Output: {OUTPUT_FILE}")
    
    # Show new columns
    new_cols = [col for col in final_df.columns if col not in df.columns]
    if new_cols:
        print(f"\nNew indicator columns ({len(new_cols)}):")
        for col in new_cols:
            print(f"  - {col}")
    
    # Check for NaN values in indicators
    print(f"\nNaN values in indicators:")
    for col in new_cols:
        nan_count = final_df[col].isna().sum()
        nan_pct = (nan_count / len(final_df)) * 100
        print(f"  {col}: {nan_count:,} ({nan_pct:.1f}%)")


if __name__ == '__main__':
    main()
