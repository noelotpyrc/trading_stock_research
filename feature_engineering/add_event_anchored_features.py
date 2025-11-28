#!/usr/bin/env python3
"""
Add event-anchored features to the consolidated earnings dataset.

Features:
1. cum_ret_since_event: Cumulative return from the event price to current close.
2. price_vs_vwap_since_event: Distance from current close to event-anchored VWAP (%).

These capture "how far has price moved since the announcement" and "is price
above or below the event VWAP", both useful for detecting overextension and
potential reversion in the first 30 minutes post-earnings.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def add_event_anchored_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute event-anchored features per ticker and event.

    Expects columns:
        - ticker
        - timestamp (datetime)
        - acceptance_datetime_utc (datetime, event time)
        - close
        - vwap_since_event (anchored VWAP, may have NaNs early in event)
        - event_price (price at event start)

    Adds columns:
        - cum_ret_since_event: (close - event_price) / event_price
        - price_vs_vwap_since_event: (close - vwap_since_event) / vwap_since_event
    """

    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if not pd.api.types.is_datetime64_any_dtype(df['acceptance_datetime_utc']):
        df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)

    # Initialize new columns
    df['cum_ret_since_event'] = np.nan
    df['price_vs_vwap_since_event'] = np.nan

    # Group by ticker + event
    grouped = df.groupby(['ticker', 'acceptance_datetime_utc'], group_keys=False)

    def compute_features(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values('timestamp').copy()

        # Event price (should be constant within event, take first non-null)
        event_price = grp['event_price'].dropna().iloc[0] if grp['event_price'].notna().any() else np.nan

        if pd.notna(event_price) and event_price > 0:
            # Cumulative return since event
            grp['cum_ret_since_event'] = (grp['close'] - event_price) / event_price

        # Price vs VWAP since event
        if 'vwap_since_event' in grp.columns:
            vwap = grp['vwap_since_event']
            grp['price_vs_vwap_since_event'] = np.where(
                vwap.notna() & (vwap > 0),
                (grp['close'] - vwap) / vwap,
                np.nan
            )

        return grp

    df = grouped.apply(compute_features)

    # Reset index if apply created a multi-index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add event-anchored features (cum_ret_since_event, price_vs_vwap_since_event)."
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
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

    # Check required columns
    required = ['ticker', 'timestamp', 'acceptance_datetime_utc', 'close', 'event_price']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Computing event-anchored features...")
    df = add_event_anchored_features(df)

    # Summary
    new_cols = ['cum_ret_since_event', 'price_vs_vwap_since_event']
    for col in new_cols:
        valid = df[col].notna().sum()
        pct = valid / len(df) * 100
        print(f"  {col}: {valid:,} valid ({pct:.1f}%)")

    print(f"Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == '__main__':
    main()

