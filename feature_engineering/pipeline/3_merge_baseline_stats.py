#!/usr/bin/env python3
"""
Merge baseline statistics into the earnings dataset.

Takes the baseline stats calculated from pre-earning periods and adds them
as additional columns to the earnings dataset using inner merge (only keep
tickers with baseline stats).
"""

import pandas as pd
from pathlib import Path

# Paths
EARNINGS_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_final.csv')
BASELINE_FILE = Path('data/baseline_stats/ticker_baseline_stats.csv')
OUTPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_baseline.csv')


def main():
    print("=" * 80)
    print("MERGING BASELINE STATISTICS INTO EARNINGS DATASET")
    print("=" * 80)
    print(f"Earnings: {EARNINGS_FILE.name}")
    print(f"Baseline: {BASELINE_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)
    print()
    
    # Load earnings dataset
    print(f"Loading earnings dataset...")
    df_earnings = pd.read_csv(EARNINGS_FILE)
    print(f"  ✓ {len(df_earnings):,} rows × {len(df_earnings.columns)} columns")
    print(f"  Tickers: {df_earnings['ticker'].nunique()}")
    
    # Load baseline stats
    print(f"Loading baseline stats...")
    df_baseline = pd.read_csv(BASELINE_FILE)
    print(f"  ✓ {len(df_baseline)} tickers with baseline stats")
    print(f"  Tickers: {', '.join(sorted(df_baseline['ticker'].tolist()))}")
    print()
    
    # Inner merge (only keep tickers with baseline stats)
    print("Merging with INNER join (only tickers with baseline stats)...")
    df_merged = df_earnings.merge(df_baseline, on='ticker', how='inner')
    
    excluded_tickers = set(df_earnings['ticker'].unique()) - set(df_baseline['ticker'].unique())
    if excluded_tickers:
        print(f"  ⚠️  Excluded {len(excluded_tickers)} tickers without baseline stats:")
        print(f"      {', '.join(sorted(excluded_tickers))}")
    
    print(f"  ✓ Merged dataset: {len(df_merged):,} rows × {len(df_merged.columns)} columns")
    print(f"  Tickers kept: {df_merged['ticker'].nunique()}")
    print(f"  Rows dropped: {len(df_earnings) - len(df_merged):,}")
    print()
    
    # Save
    print(f"Saving merged dataset...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Final dataset: {len(df_merged):,} rows × {len(df_merged.columns)} columns")
    print(f"{OUTPUT_FILE}")
    print()
    
    # Show new columns
    baseline_cols = [col for col in df_merged.columns if col.startswith('baseline_')]
    print(f"Added {len(baseline_cols)} baseline columns:")
    for col in baseline_cols:
        print(f"  • {col}")


if __name__ == '__main__':
    main()
