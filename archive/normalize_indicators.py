#!/usr/bin/env python3
"""
Normalize continuous indicators for cross-ticker modeling.

Adds relative/normalized versions of indicators:
- EMA distance from price (%)
- EMA crossover signals
- Drops raw EMA values that aren't cross-ticker comparable

Already normalized indicators (keep as-is):
- RSI (0-100 range)
- ROC (percentage)
- ATR% (normalized by price)
- vol_ratio (relative to average)
"""

import pandas as pd
from pathlib import Path

# Paths
INPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_indicators.csv')
OUTPUT_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_final.csv')


def add_normalized_features(df):
    """Add normalized/relative features from continuous indicators."""
    
    print("Adding normalized indicator features...")
    
    # EMA distance from price (%)
    # Positive = price above EMA (bullish), Negative = price below EMA (bearish)
    df['price_above_ema_short_pct'] = ((df['close'] - df['ema_short']) / df['close']) * 100
    df['price_above_ema_mid_pct'] = ((df['close'] - df['ema_mid']) / df['close']) * 100
    df['price_above_ema_long_pct'] = ((df['close'] - df['ema_long']) / df['close']) * 100
    
    # EMA crossover signals (boolean converted to 0/1)
    df['ema_short_above_mid'] = (df['ema_short'] > df['ema_mid']).astype(int)
    df['ema_mid_above_long'] = (df['ema_mid'] > df['ema_long']).astype(int)
    df['ema_bullish_alignment'] = ((df['ema_short'] > df['ema_mid']) & 
                                     (df['ema_mid'] > df['ema_long'])).astype(int)
    
    print("  ✓ Added 6 normalized EMA features")
    
    # Drop raw EMA values (not cross-ticker comparable)
    features_to_drop = ['ema_short', 'ema_mid', 'ema_long', 'atr', 'vol_ma']
    existing_drops = [f for f in features_to_drop if f in df.columns]
    
    if existing_drops:
        df = df.drop(columns=existing_drops)
        print(f"  ✓ Dropped {len(existing_drops)} ticker-specific features: {', '.join(existing_drops)}")
    
    return df


def main():
    print("=" * 80)
    print("NORMALIZING CONTINUOUS INDICATORS")
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
    
    # Add normalized features
    df = add_normalized_features(df)
    print()
    
    # Save
    print("Saving final dataset...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Show model-ready indicator columns
    indicator_cols = [col for col in df.columns if col in [
        'rsi', 'roc', 'atr_pct', 'vol_ratio',
        'price_above_ema_short_pct', 'price_above_ema_mid_pct', 'price_above_ema_long_pct',
        'ema_short_above_mid', 'ema_mid_above_long', 'ema_bullish_alignment'
    ]]
    
    print("Model-ready normalized indicators:")
    print("  Momentum:", [c for c in indicator_cols if c in ['rsi', 'roc']])
    print("  Trend:", [c for c in indicator_cols if 'ema' in c])
    print("  Volatility:", [c for c in indicator_cols if 'atr' in c])
    print("  Volume:", [c for c in indicator_cols if 'vol' in c])
    print()
    
    # Check for NaN in new features
    new_features = ['price_above_ema_short_pct', 'price_above_ema_mid_pct', 'price_above_ema_long_pct',
                    'ema_short_above_mid', 'ema_mid_above_long', 'ema_bullish_alignment']
    
    print("NaN values in new features:")
    for col in new_features:
        nan_count = df[col].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        print(f"  {col}: {nan_count:,} ({nan_pct:.1f}%)")


if __name__ == '__main__':
    main()
