# feature_engineering/add_round_lot_features.py
"""
Round Lot Features
==================

What is a Round Lot?
--------------------
A "round lot" is a second bar where:
  - num == 1: Exactly one trade occurred in that second
  - vol % 100 == 0: Volume is a multiple of 100 shares (100, 200, 300, ...)

Why Round Lots Matter:
  - Single-trade bars with round lot sizing are likely DELIBERATE human orders
  - Retail traders typically trade in round lots (100, 200 shares)
  - Institutional traders also use round lots for manual orders
  - This contrasts with HFT/algorithmic trading which produces odd lots and multiple trades per second
  - Round lots may represent "informed" or "intentional" order flow

Features
--------

SECTION 1: Rolling Volume Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These measure round lot ACTIVITY over a rolling time window.

rl_vol_{W}s
    Rolling sum of volume from round lot bars in the last W seconds.
    
    Calculation:
      1. For each bar, check if it's a round lot: (num == 1) & (vol % 100 == 0)
      2. If round lot: use vol; otherwise: use 0
      3. Sum these values over the rolling window of W seconds
      
      Pseudocode:
        rl_vol_flag = vol if is_round_lot else 0
        rl_vol_{W}s = sum(rl_vol_flag) over last W seconds
    
    Interpretation:
      - High values = lots of round lot trading activity
      - Low values = quiet period for deliberate orders
      - Spikes may indicate retail/institutional reaction to news or price levels
    
    Example: rl_vol_60s = 1500 means 1,500 shares traded via round lots in the last 60 seconds.

rl_vol_pct_{W}s
    Round lot volume as a percentage of total volume in the window.
    
    Calculation:
      1. Calculate rl_vol_{W}s (rolling round lot volume, as above)
      2. Calculate total_vol_{W}s = sum(vol) over last W seconds (all bars)
      3. Divide: rl_vol_{W}s / total_vol_{W}s * 100
      
      Pseudocode:
        rl_vol_pct_{W}s = (rl_vol_{W}s / total_vol_{W}s) * 100
    
    Interpretation:
      - High % (e.g., >50%) = market dominated by deliberate human orders
      - Low % (e.g., <10%) = market dominated by algorithmic/HFT activity
      - Changes in this ratio may signal shifts in market participant mix
    
    Example: rl_vol_pct_60s = 25.0 means 25% of volume in the last 60s came from round lots.


SECTION 2: Event-Anchored CVD Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These measure round lot ORDER FLOW DIRECTION cumulative from the earnings event.

rl_cvd_since_event
    Cumulative Volume Delta for round lots only, starting from event time.
    
    Calculation:
      1. For each bar, calculate price direction: direction = sign(vw - vw_prev)
         - vw is volume-weighted average price for the bar
         - direction = +1 if price went up, -1 if down, 0 if unchanged
      2. Calculate delta for each bar: delta = direction * vol
      3. For round lot bars only, accumulate delta from event start:
         - If round lot: add delta to cumulative sum
         - If not round lot: delta contributes 0
      4. Carry forward the cumulative sum to all bars (including non-round-lot bars)
      
      Pseudocode:
        direction = sign(vw - vw.shift(1))
        delta = direction * vol
        rl_delta = delta if is_round_lot else 0
        rl_cvd_since_event = cumsum(rl_delta) starting from event_time
    
    Interpretation:
      - Positive = round lots are net BUYING since the event
      - Negative = round lots are net SELLING since the event
      - Tracks whether deliberate human orders are accumulating or distributing

rl_cvd_zscore
    Z-score of rl_cvd_since_event within each event.
    
    Calculation:
      1. For each event, collect all rl_cvd_since_event values
      2. Calculate event_mean = mean(rl_cvd_since_event) for that event
      3. Calculate event_std = std(rl_cvd_since_event) for that event
      4. For each bar: rl_cvd_zscore = (rl_cvd_since_event - event_mean) / event_std
      
      Pseudocode:
        event_mean = mean(rl_cvd_since_event) within event
        event_std = std(rl_cvd_since_event) within event
        rl_cvd_zscore = (rl_cvd_since_event - event_mean) / event_std
    
    Interpretation:
      - Values > 2: Unusually strong round lot buying pressure
      - Values < -2: Unusually strong round lot selling pressure
      - Normalizes across events with different volatility levels


SECTION 3: Rolling CVD Ratio & Direction Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These measure round lot ORDER FLOW relative to total flow over a rolling window.

rl_cvd_ratio_{W}s
    Ratio of round lot CVD to total CVD over rolling window.
    
    Calculation:
      1. Calculate delta for each bar: delta = sign(vw - vw_prev) * vol
      2. Calculate rl_delta: delta if is_round_lot else 0
      3. Sum rl_delta over rolling window: rl_cvd_{W}s = sum(rl_delta) over last W seconds
      4. Sum delta over rolling window: total_cvd_{W}s = sum(delta) over last W seconds
      5. Divide with epsilon to avoid division by zero:
         rl_cvd_ratio_{W}s = rl_cvd_{W}s / (total_cvd_{W}s + epsilon)
      
      Pseudocode:
        delta = sign(vw - vw.shift(1)) * vol
        rl_delta = delta if is_round_lot else 0
        rl_cvd_{W}s = sum(rl_delta) over last W seconds
        total_cvd_{W}s = sum(delta) over last W seconds
        rl_cvd_ratio_{W}s = rl_cvd_{W}s / (total_cvd_{W}s + epsilon)
    
    Interpretation:
      - > 1: Round lots more directional than overall market in recent window
      - ~ 1: Round lots aligned with market
      - < 1 (but same sign): Round lots less directional than market
      - Opposite signs: Round lots trading AGAINST the market (potential reversal signal)
    
    Example: rl_cvd_ratio_60s = 1.5 means round lot flow is 50% more directional than total flow.

rl_direction_{W}s
    Simple categorical direction of round lot flow over rolling window.
    
    Calculation:
      1. Calculate rl_cvd_{W}s = sum(rl_delta) over last W seconds (as above)
      2. Apply sign function: +1 if positive, -1 if negative, 0 if zero
      
      Pseudocode:
        rl_direction_{W}s = sign(rl_cvd_{W}s)
    
    Interpretation:
      - +1: Round lots net BUYING in the recent window
      - -1: Round lots net SELLING in the recent window
      -  0: Round lots balanced (no net direction)
    
    Example: rl_direction_60s = -1 means round lots have been net selling over the last 60 seconds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def identify_round_lots(df: pd.DataFrame) -> pd.Series:
    """
    Identify round lot bars.
    
    Round lot: exactly 1 trade (num=1) with volume that is a multiple of 100.
    
    Args:
        df: DataFrame with 'num' and 'vol' columns
        
    Returns:
        Boolean Series indicating round lot bars
    """
    return (df['num'] == 1) & (df['vol'] % 100 == 0)


def compute_rolling_volume_features(event_df: pd.DataFrame, windows: list, min_periods: int) -> pd.DataFrame:
    """
    Compute rolling round lot volume features for a single event.
    
    Section 1 Features:
      - rl_vol_{W}s: Rolling sum of round lot volume
      - rl_vol_pct_{W}s: Round lot volume as % of total volume
    
    Args:
        event_df: DataFrame slice for one (ticker, event) combination
        windows: List of window sizes in seconds (e.g., [30, 60, 120])
        min_periods: Minimum observations required for rolling calculation
        
    Returns:
        DataFrame with new feature columns added
    """
    if event_df.empty:
        return event_df
    
    # Ensure sorted by timestamp and set as index for time-based rolling
    event_df = event_df.sort_values('timestamp')
    original_index = event_df.index.copy()
    event_df = event_df.set_index('timestamp')
    
    # Identify round lots
    is_round_lot = identify_round_lots(event_df)
    
    # Round lot volume (0 for non-round-lot bars)
    rl_vol_series = event_df['vol'].where(is_round_lot, 0)
    
    for window in windows:
        window_str = f'{window}s'
        
        # Feature 1: Rolling round lot volume
        event_df[f'rl_vol_{window}s'] = rl_vol_series.rolling(
            window_str, min_periods=min_periods
        ).sum()
        
        # Feature 2: Rolling round lot percentage
        # rl_vol / total_vol * 100
        rolling_total_vol = event_df['vol'].rolling(window_str, min_periods=min_periods).sum()
        rolling_rl_vol = event_df[f'rl_vol_{window}s']
        
        event_df[f'rl_vol_pct_{window}s'] = (rolling_rl_vol / rolling_total_vol * 100).replace([np.inf, -np.inf], np.nan)
    
    # Restore original index
    event_df = event_df.reset_index()
    event_df.index = original_index
    
    return event_df


def compute_event_anchored_cvd_features(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute event-anchored round lot CVD features for a single event.
    
    Section 2 Features:
      - rl_cvd_since_event: Cumulative round lot delta from event start
      - rl_cvd_zscore: Z-score of rl_cvd_since_event within the event
    
    Args:
        event_df: DataFrame slice for one (ticker, event) combination
        
    Returns:
        DataFrame with new feature columns added
    """
    if event_df.empty:
        return event_df
    
    # Ensure sorted by timestamp
    event_df = event_df.sort_values('timestamp')
    
    # Identify round lots
    is_round_lot = identify_round_lots(event_df)
    
    # Calculate price direction using vw (volume-weighted price)
    # direction = +1 if price up, -1 if price down, 0 if unchanged
    vw_diff = event_df['vw'].diff()
    direction = np.sign(vw_diff).fillna(0)
    
    # Calculate delta for each bar: direction * volume
    delta = direction * event_df['vol']
    
    # Round lot delta: only count delta from round lot bars
    rl_delta = delta.where(is_round_lot, 0).values  # Convert to numpy to avoid index issues
    
    # Feature 1: Cumulative round lot CVD since event start
    event_df['rl_cvd_since_event'] = np.cumsum(rl_delta)
    
    # Feature 2: Z-score of rl_cvd_since_event within this event
    rl_cvd = event_df['rl_cvd_since_event']
    if len(rl_cvd) > 1 and rl_cvd.std() > 0:
        event_df['rl_cvd_zscore'] = (rl_cvd - rl_cvd.mean()) / rl_cvd.std()
    else:
        event_df['rl_cvd_zscore'] = 0.0
    
    return event_df


def compute_rolling_cvd_features(event_df: pd.DataFrame, windows: list, min_periods: int) -> pd.DataFrame:
    """
    Compute rolling round lot CVD ratio and direction features for a single event.
    
    Section 3 Features:
      - rl_cvd_ratio_{W}s: Round lot CVD / Total CVD over rolling window
      - rl_direction_{W}s: Sign of round lot CVD over rolling window (+1/-1/0)
    
    Args:
        event_df: DataFrame slice for one (ticker, event) combination
        windows: List of window sizes in seconds (e.g., [30, 60, 120])
        min_periods: Minimum observations required for rolling calculation
        
    Returns:
        DataFrame with new feature columns added
    """
    if event_df.empty:
        return event_df
    
    # Ensure sorted by timestamp and set as index for time-based rolling
    event_df = event_df.sort_values('timestamp')
    original_index = event_df.index.copy()
    event_df = event_df.set_index('timestamp')
    
    # Identify round lots
    is_round_lot = identify_round_lots(event_df)
    
    # Calculate price direction using vw (volume-weighted price)
    vw_diff = event_df['vw'].diff()
    direction = np.sign(vw_diff).fillna(0)
    
    # Calculate delta for each bar: direction * volume
    delta = direction * event_df['vol']
    
    # Round lot delta: only count delta from round lot bars
    rl_delta = delta.where(is_round_lot, 0)
    
    epsilon = 1e-8  # Small value to avoid division by zero
    
    for window in windows:
        window_str = f'{window}s'
        
        # Rolling CVD for round lots
        rl_cvd_rolling = rl_delta.rolling(window_str, min_periods=min_periods).sum()
        
        # Rolling CVD for all bars (total)
        total_cvd_rolling = delta.rolling(window_str, min_periods=min_periods).sum()
        
        # Feature 1: Round lot CVD ratio
        # rl_cvd / (total_cvd + epsilon)
        event_df[f'rl_cvd_ratio_{window}s'] = rl_cvd_rolling / (total_cvd_rolling + epsilon)
        # Handle cases where total_cvd is very small (ratio becomes huge)
        event_df[f'rl_cvd_ratio_{window}s'] = event_df[f'rl_cvd_ratio_{window}s'].replace([np.inf, -np.inf], np.nan)
        
        # Feature 2: Round lot direction
        # sign(rl_cvd_rolling): +1 buying, -1 selling, 0 neutral
        event_df[f'rl_direction_{window}s'] = np.sign(rl_cvd_rolling)
    
    # Restore original index
    event_df = event_df.reset_index()
    event_df.index = original_index
    
    return event_df


def main():
    parser = argparse.ArgumentParser(description="Add round lot features to a consolidated CSV.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input consolidated CSV file.")
    parser.add_argument("--output", type=str, help="Path to the output CSV file. If not provided, overwrites input.", default=None)
    parser.add_argument("--windows", type=str, default="30,60,120", help="Comma-separated list of rolling windows in seconds.")
    parser.add_argument("--min-periods", type=int, default=10, help="Minimum observations in window to produce a value.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    windows = [int(w) for w in args.windows.split(',')]
    min_periods = args.min_periods

    print("=" * 70)
    print("ROUND LOT FEATURES")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Windows: {windows}")
    print(f"Min periods: {min_periods}")
    print("=" * 70)
    print()

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print()

    # Check for required columns
    required_cols = ['num', 'vol', 'vw', 'timestamp', 'ticker', 'acceptance_datetime_utc']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Count round lots
    is_rl = identify_round_lots(df)
    print(f"Round lot bars: {is_rl.sum():,} ({is_rl.mean()*100:.1f}% of all bars)")
    print()

    initial_columns = set(df.columns)

    # Group by ticker and event for per-event calculations
    grouped = df.groupby(['ticker', 'acceptance_datetime_utc'], group_keys=False)

    # ===== SECTION 1: Rolling Volume Features =====
    print("-" * 70)
    print("SECTION 1: Rolling Volume Features")
    print("-" * 70)
    print(f"  Features: rl_vol_{{W}}s, rl_vol_pct_{{W}}s")
    print(f"  Windows: {windows}")
    df = grouped.apply(lambda x: compute_rolling_volume_features(x, windows, min_periods))
    print("  ✓ Done")
    print()

    # Re-group after applying (groupby object is consumed)
    grouped = df.groupby(['ticker', 'acceptance_datetime_utc'], group_keys=False)

    # ===== SECTION 2: Event-Anchored CVD Features =====
    print("-" * 70)
    print("SECTION 2: Event-Anchored CVD Features")
    print("-" * 70)
    print(f"  Features: rl_cvd_since_event, rl_cvd_zscore")
    df = grouped.apply(compute_event_anchored_cvd_features)
    print("  ✓ Done")
    print()

    # Re-group after applying
    grouped = df.groupby(['ticker', 'acceptance_datetime_utc'], group_keys=False)

    # ===== SECTION 3: Rolling CVD Ratio & Direction Features =====
    print("-" * 70)
    print("SECTION 3: Rolling CVD Ratio & Direction Features")
    print("-" * 70)
    print(f"  Features: rl_cvd_ratio_{{W}}s, rl_direction_{{W}}s")
    print(f"  Windows: {windows}")
    df = grouped.apply(lambda x: compute_rolling_cvd_features(x, windows, min_periods))
    print("  ✓ Done")
    print()

    # Report new columns
    final_columns = set(df.columns)
    new_columns = sorted(list(final_columns - initial_columns))

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Initial columns: {len(initial_columns)}")
    print(f"Final columns:   {len(final_columns)}")
    print(f"New columns:     {len(new_columns)}")
    print()

    print("New columns added:")
    for col in new_columns:
        valid_count = df[col].count()
        pct_valid = (valid_count / len(df)) * 100
        print(f"  • {col}: {valid_count:,} valid ({pct_valid:.1f}%)")
    print()

    # Save
    print(f"Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("  ✓ Done")
    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

