#!/usr/bin/env python3
"""
Orchestrator script to generate all new features and merge them into a single enriched dataset.

Applies:
1. Event-anchored features (cum_ret_since_event, price_vs_vwap_since_event)
2. Windowed features (vol_intensity, range_expansion, cvd_slope, etc.) at 30s, 60s, 120s
3. Momentum features (cum_ret_zscore, vol_surge_ratio, trade_intensity_zscore)

Input: earnings_consolidated_final_with_10_20s.csv (already has 10s/20s targets)
Output: earnings_consolidated_enriched.csv
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering.add_event_anchored_features import add_event_anchored_features
from feature_engineering.add_windowed_features import add_windowed_features
from feature_engineering.add_momentum_features import add_momentum_features


# Default paths
DEFAULT_INPUT = '/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_final.csv'
DEFAULT_OUTPUT = '/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_enriched.csv'

# Feature parameters
WINDOWED_WINDOWS = [30, 60, 120]
MOMENTUM_CUM_RET_WINDOWS = [5, 10, 20]
MOMENTUM_BASELINE_WINDOWS = [30, 60, 120]
MIN_PERIODS = 10


def main():
    parser = argparse.ArgumentParser(
        description="Generate all new features and create enriched dataset."
    )
    parser.add_argument(
        '--input',
        type=str,
        default=DEFAULT_INPUT,
        help=f'Path to input CSV (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        help=f'Path to output CSV (default: {DEFAULT_OUTPUT})'
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 80)
    print("GENERATING ALL NEW FEATURES")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading dataset...")
    df = pd.read_csv(input_path)
    initial_cols = len(df.columns)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {initial_cols}")
    print()

    # --- 1. Event-Anchored Features ---
    print("-" * 80)
    print("1. EVENT-ANCHORED FEATURES")
    print("-" * 80)
    print("  Adding: cum_ret_since_event, price_vs_vwap_since_event")
    df = add_event_anchored_features(df)
    print(f"  ✓ Done")
    print()

    # --- 2. Windowed Features ---
    print("-" * 80)
    print("2. WINDOWED FEATURES")
    print("-" * 80)
    print(f"  Windows: {WINDOWED_WINDOWS}")
    print("  Features: vol_intensity, range_expansion, cvd_slope, close_position_avg,")
    print("            new_high_count, new_low_count, vol_decay_rate")

    for window in WINDOWED_WINDOWS:
        df = add_windowed_features(df, window=window, min_periods=MIN_PERIODS)

    print(f"  ✓ Done")
    print()

    # --- 3. Momentum Features ---
    print("-" * 80)
    print("3. MOMENTUM FEATURES")
    print("-" * 80)
    print(f"  Cumulative return windows: {MOMENTUM_CUM_RET_WINDOWS}")
    print(f"  Baseline windows: {MOMENTUM_BASELINE_WINDOWS}")
    print("  Features: cum_ret_zscore, vol_surge_ratio, trade_intensity_zscore")

    for cum_ret_window in MOMENTUM_CUM_RET_WINDOWS:
        for baseline_window in MOMENTUM_BASELINE_WINDOWS:
            df = add_momentum_features(
                df,
                cum_ret_window=cum_ret_window,
                baseline_window=baseline_window,
                min_periods=MIN_PERIODS
            )

    print(f"  ✓ Done")
    print()

    # --- Summary ---
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    final_cols = len(df.columns)
    new_cols = final_cols - initial_cols

    print(f"Initial columns: {initial_cols}")
    print(f"Final columns:   {final_cols}")
    print(f"New columns:     {new_cols}")
    print()

    # List new columns by category
    new_col_names = list(df.columns)[initial_cols:]

    event_anchored = [c for c in new_col_names if c in ['cum_ret_since_event', 'price_vs_vwap_since_event']]
    windowed = [c for c in new_col_names if any(c.startswith(p) for p in [
        'vol_intensity_', 'range_expansion_', 'cvd_slope_', 'close_position_avg_',
        'new_high_count_', 'new_low_count_', 'vol_decay_rate_'
    ])]
    momentum = [c for c in new_col_names if any(c.startswith(p) for p in [
        'cum_ret_', 'vol_surge_ratio_', 'trade_intensity_zscore_'
    ]) and c not in event_anchored]

    print(f"Event-anchored features ({len(event_anchored)}):")
    for col in sorted(event_anchored):
        valid = df[col].notna().sum()
        pct = valid / len(df) * 100
        print(f"  • {col}: {valid:,} valid ({pct:.1f}%)")

    print(f"\nWindowed features ({len(windowed)}):")
    for col in sorted(windowed):
        valid = df[col].notna().sum()
        pct = valid / len(df) * 100
        print(f"  • {col}: {valid:,} valid ({pct:.1f}%)")

    print(f"\nMomentum features ({len(momentum)}):")
    for col in sorted(momentum):
        valid = df[col].notna().sum()
        pct = valid / len(df) * 100
        print(f"  • {col}: {valid:,} valid ({pct:.1f}%)")

    # Save
    print()
    print("-" * 80)
    print(f"Saving enriched dataset to: {output_path}")
    df.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  File size: {file_size_mb:.1f} MB")
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()

