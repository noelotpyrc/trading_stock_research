#!/usr/bin/env python3
"""
Add derived features needed for strategy signal generation.

Adds:
1. CVD z-score (per-event normalization)
2. Rolling delta sums (20s, 10s windows)
3. Seconds since earnings (helper column)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
INPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_final.csv')
OUTPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_derived.csv')


def add_derived_features(df):
    """Add derived features for signal generation."""
    
    print("Adding derived features...")
    
    # 1. Seconds since earnings
    print("  Calculating seconds_since_event...")
    df['seconds_since_event'] = (
        pd.to_datetime(df['timestamp']) -
        pd.to_datetime(df['acceptance_datetime_utc'])
    ).dt.total_seconds()
    
    # 2. Per-event CVD z-score and rolling deltas
    print(f"  Processing {df['acceptance_datetime_utc'].nunique()} events...")
    
    # Process each event separately
    events = df['acceptance_datetime_utc'].unique()
    results = []
    
    for i, event_time in enumerate(events, 1):
        if i % 50 == 0:
            print(f"    Event {i}/{len(events)}")
        
        event_mask = df['acceptance_datetime_utc'] == event_time
        event_df = df[event_mask].copy()
        
        # Sort by timestamp
        event_df = event_df.sort_values('timestamp')
        
        # CVD z-score within this event
        cvd = event_df['cvd_since_event']
        if len(cvd) > 1 and cvd.std() > 0:
            event_df['cvd_zscore'] = (cvd - cvd.mean()) / cvd.std()
        else:
            event_df['cvd_zscore'] = 0.0
        
        # Rolling delta sums (need to calculate delta first)
        # Delta = change in CVD
        event_df['delta'] = event_df['cvd_since_event'].diff().fillna(0)
        
        # Rolling delta sums (20s and 10s windows)
        event_df['delta_20s'] = event_df['delta'].rolling(window=20, min_periods=1).sum()
        event_df['delta_10s'] = event_df['delta'].rolling(window=10, min_periods=1).sum()
        
        results.append(event_df)
    
    # Combine all events
    final_df = pd.concat(results, ignore_index=True)
    
    # Sort back by ticker and timestamp
    final_df = final_df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    print(f"  ✓ Added 5 derived features")
    return final_df


def main():
    print("=" * 80)
    print("ADDING DERIVED FEATURES FOR SIGNAL GENERATION")
    print("=" * 80)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)
    print()
    
    # Load dataset
    print(f"Loading {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print()
    
    # Add derived features
    df = add_derived_features(df)
    print()
    
    # Save
    print("Saving dataset with derived features...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)} (was 56, added 5)")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Show new features
    new_features = ['seconds_since_event', 'cvd_zscore', 'delta', 'delta_20s', 'delta_10s']
    print("New derived features:")
    for feat in new_features:
        if feat in df.columns:
            print(f"  ✓ {feat}")
            nan_count = df[feat].isna().sum()
            print(f"    NaN: {nan_count:,} ({nan_count/len(df)*100:.1f}%)")
            if df[feat].dtype in ['float64', 'int64']:
                print(f"    Range: [{df[feat].min():.2f}, {df[feat].max():.2f}]")


if __name__ == '__main__':
    main()
