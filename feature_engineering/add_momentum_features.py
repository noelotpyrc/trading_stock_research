#!/usr/bin/env python3
"""
Add momentum-based features to the consolidated earnings dataset.

Features (all with configurable windows):
1. cum_ret_Xs_zscore_Ys: Z-score of cumulative return over X seconds, baselined over Y seconds
2. vol_surge_ratio_Ys: Current volume / rolling mean volume over Y seconds
3. trade_intensity_zscore_Ys: Z-score of trade count over Y seconds

These capture unusual price momentum, volume surges, and trade activity spikes
that are predictive for short-horizon (10-20s) returns in volatile post-earnings periods.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def add_momentum_features(
    df: pd.DataFrame,
    cum_ret_window: int = 10,
    baseline_window: int = 60,
    min_periods: int = 10
) -> pd.DataFrame:
    """
    Compute momentum-based features per ticker and event.

    Args:
        df: DataFrame with OHLCV data and event metadata
        cum_ret_window: Window for cumulative return calculation (default 10s)
        baseline_window: Window for rolling baseline (mean/std) calculation (default 60s)
        min_periods: Minimum periods required for rolling calculations

    Expects columns:
        - ticker
        - timestamp (datetime)
        - acceptance_datetime_utc (datetime, event time)
        - close, vol, num

    Adds columns:
        - cum_ret_{cum_ret_window}s_zscore_{baseline_window}s
        - vol_surge_ratio_{baseline_window}s
        - trade_intensity_zscore_{baseline_window}s
    """

    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if not pd.api.types.is_datetime64_any_dtype(df['acceptance_datetime_utc']):
        df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)

    # Column names
    col_cum_ret_zscore = f'cum_ret_{cum_ret_window}s_zscore_{baseline_window}s'
    col_vol_surge = f'vol_surge_ratio_{baseline_window}s'
    col_trade_intensity = f'trade_intensity_zscore_{baseline_window}s'

    # Initialize new columns
    for col in [col_cum_ret_zscore, col_vol_surge, col_trade_intensity]:
        df[col] = np.nan

    # Group by ticker + event
    grouped = df.groupby(['ticker', 'acceptance_datetime_utc'], group_keys=False)

    def compute_features(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values('timestamp').copy()
        n = len(grp)

        if n < min_periods:
            return grp

        # --- 1. Cumulative Return Z-Score ---
        # Cumulative return over cum_ret_window bars
        cum_ret = (grp['close'] - grp['close'].shift(cum_ret_window)) / grp['close'].shift(cum_ret_window)

        # Z-score relative to baseline_window
        cum_ret_mean = cum_ret.rolling(window=baseline_window, min_periods=min_periods).mean()
        cum_ret_std = cum_ret.rolling(window=baseline_window, min_periods=min_periods).std()
        grp[col_cum_ret_zscore] = (cum_ret - cum_ret_mean) / cum_ret_std.replace(0, np.nan)

        # --- 2. Volume Surge Ratio ---
        # Current volume / rolling mean volume
        vol_mean = grp['vol'].rolling(window=baseline_window, min_periods=min_periods).mean()
        grp[col_vol_surge] = grp['vol'] / vol_mean.replace(0, np.nan)

        # --- 3. Trade Intensity Z-Score ---
        # Z-score of trade count (num)
        if 'num' in grp.columns:
            num_mean = grp['num'].rolling(window=baseline_window, min_periods=min_periods).mean()
            num_std = grp['num'].rolling(window=baseline_window, min_periods=min_periods).std()
            grp[col_trade_intensity] = (grp['num'] - num_mean) / num_std.replace(0, np.nan)

        return grp

    print(f"  Computing momentum features (cum_ret={cum_ret_window}s, baseline={baseline_window}s)...")
    df = grouped.apply(compute_features)

    # Reset index if apply created a multi-index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add momentum-based features with configurable windows."
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_final.csv',
        help='Path to input CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output CSV (default: overwrite input)'
    )
    parser.add_argument(
        '--cum-ret-windows',
        type=str,
        default='10',
        help='Comma-separated cumulative return windows in seconds (default: 10)'
    )
    parser.add_argument(
        '--baseline-windows',
        type=str,
        default='60',
        help='Comma-separated baseline windows in seconds (default: 60)'
    )
    parser.add_argument(
        '--min-periods',
        type=int,
        default=10,
        help='Minimum periods for rolling calculations (default: 10)'
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    cum_ret_windows = [int(w.strip()) for w in args.cum_ret_windows.split(',')]
    baseline_windows = [int(w.strip()) for w in args.baseline_windows.split(',')]

    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

    # Check required columns
    required = ['ticker', 'timestamp', 'acceptance_datetime_utc', 'close', 'vol']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Computing momentum features:")
    print(f"  Cumulative return windows: {cum_ret_windows}")
    print(f"  Baseline windows: {baseline_windows}")

    # Compute for all combinations of cum_ret_window × baseline_window
    for cum_ret_window in cum_ret_windows:
        for baseline_window in baseline_windows:
            df = add_momentum_features(
                df,
                cum_ret_window=cum_ret_window,
                baseline_window=baseline_window,
                min_periods=args.min_periods
            )

    # Summary of new columns
    new_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in [
            'cum_ret_', 'vol_surge_ratio_', 'trade_intensity_zscore_'
        ]
    )]
    print(f"\nAdded {len(new_cols)} new columns:")
    for col in sorted(new_cols):
        valid = df[col].notna().sum()
        pct = valid / len(df) * 100
        print(f"  {col}: {valid:,} valid ({pct:.1f}%)")

    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == '__main__':
    main()


