#!/usr/bin/env python3
"""
Generate ADAPTIVE trading signals using baseline statistics.

This script uses ticker-specific baseline stats to set adaptive thresholds,
avoiding the problem where high-volatility tickers have unreachable thresholds.

Approach: mean + n*std instead of percentiles
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from strategies.signal_generators_adaptive import generate_adaptive_signals

# Paths
INPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_baseline.csv')
OUTPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_with_adaptive_signals.csv')


def main():
    print("=" * 80)
    print("GENERATING ADAPTIVE STRATEGY SIGNALS")
    print("=" * 80)
    print(f"Input:  {INPUT_FILE.name}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)
    print()
    
    # Load dataset with baseline stats
    print(f"Loading {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Verify baseline columns exist
    baseline_cols = [col for col in df.columns if col.startswith('baseline_')]
    print(f"  Found {len(baseline_cols)} baseline stat columns")
    print()
    
    # Generate adaptive signals
    print("Generating ADAPTIVE signals using baseline stats...")
    print("  Strategy 1: EIR (volume z-score)")
    print("  Strategy 1.1: Impulse (mean + 1.5×std range)")
    print("  Strategy 2: CVD (already adaptive)")
    print("  Strategy 3: Fade (all thresholds adaptive)")
    print()
    
    df = generate_adaptive_signals(df)
    
    # Count signals
    print("Adaptive signal counts:")
    signal_cols = [col for col in df.columns if col.startswith('signal_') and 'adaptive' in col]
    for col in sorted(signal_cols):
        count = df[col].sum()
        pct = (count / len(df)) * 100
        print(f"  {col:35s}: {count:6,} ({pct:5.2f}%)")
    print()
    
    # Save
    print("Saving dataset with adaptive signals...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)} (was 79, added {len(signal_cols)})")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Compare by strategy
    print("Signals by strategy (ADAPTIVE):")
    strategies = ['eir', 'impulse', 'cvd', 'fade']
    for strat in strategies:
        long_col = f'signal_{strat}_adaptive_long'
        short_col = f'signal_{strat}_adaptive_short'
        
        if long_col in df.columns and short_col in df.columns:
            long_count = df[long_col].sum()
            short_count = df[short_col].sum()
            total = long_count + short_count
            print(f"  {strat.upper():10s}: {total:6,} total ({long_count:,} long, {short_count:,} short)")


if __name__ == '__main__':
    main()
