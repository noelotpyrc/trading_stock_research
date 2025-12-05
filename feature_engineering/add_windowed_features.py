#!/usr/bin/env python3
"""
Add windowed features to the consolidated earnings dataset.

Features (all computed with configurable lookback windows):
1. vol_intensity: Current volume relative to rolling mean (volume surge detection)
2. range_expansion: Current bar range relative to rolling mean range (volatility burst)
3. cvd_slope: Rate of change of CVD over the window (flow acceleration/deceleration)
4. close_position_avg: Average close position within bar over window (sustained momentum)
5. new_high_count: Count of new local highs in window (trending vs consolidating)
6. new_low_count: Count of new local lows in window
7. vol_decay_rate: Ratio of recent volume to earlier volume in window (volume fading)

All features are computed per ticker + event to avoid cross-event contamination.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def compute_close_position(df: pd.DataFrame) -> pd.Series:
    """
    Compute where close sits within the bar's range.
    Returns value between 0 (closed at low) and 1 (closed at high).
    """
    bar_range = df['high'] - df['low']
    return np.where(
        bar_range > 0,
        (df['close'] - df['low']) / bar_range,
        0.5  # If no range, neutral
    )


def compute_bar_range_pct(df: pd.DataFrame) -> pd.Series:
    """Compute bar range as percentage of close price."""
    return (df['high'] - df['low']) / df['close']


def add_windowed_features(
    df: pd.DataFrame,
    window: int = 60,
    min_periods: int = 10
) -> pd.DataFrame:
    """
    Compute windowed features per ticker and event.

    Args:
        df: DataFrame with OHLCV data and event metadata
        window: Lookback window in seconds/bars (default 60)
        min_periods: Minimum periods required for rolling calculations

    Expects columns:
        - ticker
        - timestamp (datetime)
        - acceptance_datetime_utc (datetime, event time)
        - open, high, low, close, vol
        - cvd_since_event (optional, for cvd_slope)

    Adds columns (suffixed with window size):
        - vol_intensity_{window}s
        - range_expansion_{window}s
        - cvd_slope_{window}s
        - close_position_avg_{window}s
        - new_high_count_{window}s
        - new_low_count_{window}s
        - vol_decay_rate_{window}s
    """

    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if not pd.api.types.is_datetime64_any_dtype(df['acceptance_datetime_utc']):
        df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)

    # Column names with window suffix
    col_vol_intensity = f'vol_intensity_{window}s'
    col_range_expansion = f'range_expansion_{window}s'
    col_cvd_slope = f'cvd_slope_{window}s'
    col_close_pos_avg = f'close_position_avg_{window}s'
    col_new_high = f'new_high_count_{window}s'
    col_new_low = f'new_low_count_{window}s'
    col_vol_decay = f'vol_decay_rate_{window}s'

    # Initialize new columns
    for col in [col_vol_intensity, col_range_expansion, col_cvd_slope,
                col_close_pos_avg, col_new_high, col_new_low, col_vol_decay]:
        df[col] = np.nan

    # Group by ticker + event
    grouped = df.groupby(['ticker', 'acceptance_datetime_utc'], group_keys=False)

    def compute_features(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values('timestamp').copy()
        n = len(grp)

        if n < min_periods:
            return grp

        # --- 1. Volume Intensity ---
        # Current volume / rolling mean volume
        vol_mean = grp['vol'].rolling(window=window, min_periods=min_periods).mean()
        grp[col_vol_intensity] = grp['vol'] / vol_mean.replace(0, np.nan)

        # --- 2. Range Expansion ---
        # Current bar range % / rolling mean bar range %
        bar_range_pct = compute_bar_range_pct(grp)
        range_mean = bar_range_pct.rolling(window=window, min_periods=min_periods).mean()
        grp[col_range_expansion] = bar_range_pct / range_mean.replace(0, np.nan)

        # --- 3. CVD Slope ---
        # Rate of change of CVD over the window (linear slope approximation)
        if 'cvd_since_event' in grp.columns:
            cvd = grp['cvd_since_event']
            # Slope = (current - value at start of window) / window
            cvd_lagged = cvd.shift(window)
            grp[col_cvd_slope] = (cvd - cvd_lagged) / window

        # --- 4. Close Position Average ---
        # Average of close_position over the window (sustained buying/selling)
        close_pos = compute_close_position(grp)
        grp[col_close_pos_avg] = pd.Series(close_pos, index=grp.index).rolling(
            window=window, min_periods=min_periods
        ).mean()

        # --- 5 & 6. New High/Low Count ---
        # Count how many times we made a new rolling high/low in the window
        rolling_max = grp['high'].rolling(window=window, min_periods=1).max()
        rolling_min = grp['low'].rolling(window=window, min_periods=1).min()

        # A new high occurs when current high equals the rolling max
        is_new_high = (grp['high'] >= rolling_max.shift(1)).astype(float)
        is_new_low = (grp['low'] <= rolling_min.shift(1)).astype(float)

        grp[col_new_high] = is_new_high.rolling(window=window, min_periods=min_periods).sum()
        grp[col_new_low] = is_new_low.rolling(window=window, min_periods=min_periods).sum()

        # --- 7. Volume Decay Rate ---
        # Ratio of volume in recent half of window vs first half
        # High value = volume still strong; low value = volume fading
        half_window = max(window // 2, 1)
        vol_recent = grp['vol'].rolling(window=half_window, min_periods=min_periods // 2).sum()
        vol_earlier = grp['vol'].shift(half_window).rolling(
            window=half_window, min_periods=min_periods // 2
        ).sum()
        grp[col_vol_decay] = vol_recent / vol_earlier.replace(0, np.nan)

        return grp

    print(f"  Computing features with {window}s window...")
    df = grouped.apply(compute_features)

    # Reset index if apply created a multi-index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add windowed features with configurable lookback."
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
        '--windows',
        type=str,
        default='60,120',
        help='Comma-separated list of window sizes in seconds (default: 60,120)'
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
    windows = [int(w.strip()) for w in args.windows.split(',')]

    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

    # Check required columns
    required = ['ticker', 'timestamp', 'acceptance_datetime_utc',
                'open', 'high', 'low', 'close', 'vol']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Computing windowed features for windows: {windows}")
    for window in windows:
        df = add_windowed_features(df, window=window, min_periods=args.min_periods)

    # Summary of new columns
    new_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in [
            'vol_intensity_', 'range_expansion_', 'cvd_slope_',
            'close_position_avg_', 'new_high_count_', 'new_low_count_', 'vol_decay_rate_'
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


