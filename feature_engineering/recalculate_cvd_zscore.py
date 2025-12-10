#!/usr/bin/env python3
"""
Recalculate CVD Z-Score using correct expanding window method.

The original cvd_zscore was calculated using full-event mean/std (look-ahead bias).
The correct calculation uses expanding window of CVD values accumulated SINCE the event:
    z_cvd = (current_cvd - mean(cvd_history)) / std(cvd_history)

Where cvd_history grows each second from event_time onwards.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering.indicators import estimate_delta

# Paths
INPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_0_to_120m_consolidated.csv')
OUTPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_0_to_120m_consolidated_fixed_zscore.csv')


def calculate_cvd_zscore_expanding_per_event(event_df, min_periods=60):
    """
    Calculate CVD Z-Score using expanding window for a single event.
    
    Args:
        event_df: DataFrame for one event, sorted by timestamp
        min_periods: Minimum samples before z-score is valid (default 60)
    
    Returns:
        Series: CVD z-score values
    """
    # Get cvd_since_event
    if 'cvd_since_event' not in event_df.columns:
        return pd.Series(np.nan, index=event_df.index)
    
    cvd = event_df['cvd_since_event'].values
    n = len(cvd)
    
    # Initialize result
    z_scores = np.full(n, np.nan)
    
    # For each bar, calculate z-score using all CVD values up to that point
    for i in range(min_periods, n):
        cvd_history = cvd[:i+1]  # All CVD values from start to current bar
        
        mean_cvd = np.nanmean(cvd_history)
        std_cvd = np.nanstd(cvd_history)
        
        if std_cvd > 1e-10:
            z_scores[i] = (cvd[i] - mean_cvd) / std_cvd
    
    return pd.Series(z_scores, index=event_df.index)


def main():
    print("=" * 80)
    print("RECALCULATING CVD Z-SCORE WITH EXPANDING WINDOW")
    print("=" * 80)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)
    print()
    
    # Check input exists
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Ensure timestamps are datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)
    
    # Sort by ticker, event, timestamp
    df = df.sort_values(['ticker', 'acceptance_datetime_utc', 'timestamp']).reset_index(drop=True)
    
    # Store original for comparison
    if 'cvd_zscore' in df.columns:
        df['cvd_zscore_old'] = df['cvd_zscore'].copy()
        old_stats = df['cvd_zscore'].describe()
        print(f"\nOriginal cvd_zscore stats:")
        print(f"  Mean: {old_stats['mean']:.4f}")
        print(f"  Std:  {old_stats['std']:.4f}")
        print(f"  NaN:  {df['cvd_zscore'].isna().sum():,} ({df['cvd_zscore'].isna().mean()*100:.1f}%)")
    
    # Recalculate cvd_zscore per event
    print("\nRecalculating cvd_zscore with expanding window...")
    
    # Group by ticker and event
    grouped = df.groupby(['ticker', 'acceptance_datetime_utc'], group_keys=False)
    n_events = grouped.ngroups
    print(f"  Processing {n_events} events...")
    
    # Process each event
    new_zscore = []
    processed = 0
    
    for (ticker, event_time), event_df in grouped:
        event_df_sorted = event_df.sort_values('timestamp')
        z_scores = calculate_cvd_zscore_expanding_per_event(event_df_sorted, min_periods=60)
        new_zscore.append(z_scores)
        
        processed += 1
        if processed % 100 == 0:
            print(f"    {processed}/{n_events} events processed...")
    
    # Combine results
    df['cvd_zscore'] = pd.concat(new_zscore)
    
    # Show new stats
    new_stats = df['cvd_zscore'].describe()
    print(f"\nNew cvd_zscore stats:")
    print(f"  Mean: {new_stats['mean']:.4f}")
    print(f"  Std:  {new_stats['std']:.4f}")
    print(f"  NaN:  {df['cvd_zscore'].isna().sum():,} ({df['cvd_zscore'].isna().mean()*100:.1f}%)")
    
    # Compare old vs new
    if 'cvd_zscore_old' in df.columns:
        correlation = df[['cvd_zscore', 'cvd_zscore_old']].corr().iloc[0, 1]
        print(f"\nCorrelation between old and new: {correlation:.4f}")
        
        # Drop the old column before saving
        df = df.drop(columns=['cvd_zscore_old'])
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 ** 2)
    print(f"  File size: {file_size_mb:.1f} MB")
    
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
