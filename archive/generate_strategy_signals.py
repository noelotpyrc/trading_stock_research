#!/usr/bin/env python3
"""
Generate trading signals for all strategies and add to dataset.

Uses the lightweight functional signal generators to create binary
entry signals for:
1. EIR (Opening Range Breakout)
2. Impulse Bar Breakout
3. CVD Momentum
4. Overreaction Fade
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from strategies.signal_generators import generate_all_signals

# Paths
INPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_derived.csv')
OUTPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_signals.csv')


def main():
    print("=" * 80)
    print("GENERATING STRATEGY SIGNALS")
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
    
    # Generate signals with default parameters
    print("Generating signals for all strategies...")
    print("  Strategy 1: EIR (Opening Range Breakout)")
    print("  Strategy 1.1: Impulse Bar Breakout")
    print("  Strategy 2: CVD Momentum")
    print("  Strategy 3: Overreaction Fade")
    print()
    
    df = generate_all_signals(df)
    
    # Count signals
    print("Signal counts:")
    signal_cols = [col for col in df.columns if col.startswith('signal_')]
    for col in sorted(signal_cols):
        count = df[col].sum()
        pct = (count / len(df)) * 100
        print(f"  {col:25s}: {count:6,} ({pct:5.2f}%)")
    print()
    
    # Save
    print("Saving dataset with signals...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)} (was 61, added {len(signal_cols)})")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Show signal summary by strategy
    print("Signals by strategy:")
    strategies = ['eir', 'impulse', 'cvd', 'fade']
    for strat in strategies:
        long_col = f'signal_{strat}_long'
        short_col = f'signal_{strat}_short'
        
        if long_col in df.columns and short_col in df.columns:
            long_count = df[long_col].sum()
            short_count = df[short_col].sum()
            total = long_count + short_count
            print(f"  {strat.upper():10s}: {total:6,} total ({long_count:,} long, {short_count:,} short)")


if __name__ == '__main__':
    main()
