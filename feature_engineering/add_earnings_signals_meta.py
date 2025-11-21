import pandas as pd
import numpy as np
import argparse
import os
from feature_engineering.label_earning_event_windows import add_features as label_windows
from feature_engineering.indicators import (
    calculate_cvd, calculate_anchored_vwap
)
from feature_engineering.microstructure import (
    get_opening_range, detect_impulse_bar
)
from feature_engineering.windows import get_time_window_mask
from feature_engineering.targets import (
    get_forward_returns, get_max_excursion, get_hit_target_pct, get_first_touch_target
)

def add_earnings_signals(input_file, output_file, pre_labeled=True):
    """
    Calculates event-specific metadata and targets for earnings strategies.
    Outputs ONLY rows within event windows.
    
    Args:
        input_file: Path to input CSV (either raw OHLCV or pre-labeled)
        output_file: Path to output CSV
        pre_labeled: If True, assumes input has window labels. If False, labels on the fly.
    """
    print(f"Processing {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Ensure timestamps are datetime UTC
    if 'timestamp' not in df.columns:
        print("Error: 'timestamp' column not found.")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    if 'acceptance_datetime_utc' not in df.columns:
        print("Error: 'acceptance_datetime_utc' column not found.")
        return
        
    df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)

    # Label windows if not pre-labeled
    if not pre_labeled or 'event_plus_minus_30m' not in df.columns:
        print("Labeling event windows...")
        # Save to temp file, then reload (simpler than refactoring label_windows to return df)
        temp_labeled = input_file.replace('.csv', '_temp_labeled.csv')
        label_windows(input_file, temp_labeled)
        df = pd.read_csv(temp_labeled)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)
        os.remove(temp_labeled)

    # --- Event-Specific Signals (Batch Processing) ---
    print("Calculating event-specific signals...")
    
    # Initialize columns
    event_cols = [
        'cvd_since_event', 'vwap_since_event', 
        'or_high', 'or_low', 
        'event_price',
        'impulse_high', 'impulse_low',
        'first_5m_high', 'first_5m_low', 'peak_1m_vol_5m',
        'vwap_uptime'
    ]
    
    for col in event_cols:
        df[col] = np.nan

    unique_events = df['acceptance_datetime_utc'].unique()
    print(f"Found {len(unique_events)} unique events.")

    for event_time in unique_events:
        if pd.isna(event_time):
            continue
            
        event_mask = df['acceptance_datetime_utc'] == event_time
        if not event_mask.any():
            continue
            
        # Get the slice
        event_slice = df.loc[event_mask].copy()
        event_slice = event_slice.sort_values('timestamp')
        
        # CVD Since Event
        cvd_series = calculate_cvd(event_slice, start_time=event_time)
        df.loc[event_mask, 'cvd_since_event'] = cvd_series
        
        # Anchored VWAP Since Event
        vwap_series = calculate_anchored_vwap(event_slice, start_time=event_time)
        df.loc[event_mask, 'vwap_since_event'] = vwap_series
        
        # Opening Range (First 60s after event)
        or_high, or_low = get_opening_range(event_slice, event_time, duration_seconds=60)
        df.loc[event_mask, 'or_high'] = or_high
        df.loc[event_mask, 'or_low'] = or_low
        
        # Event Price
        if not event_slice.empty:
            event_price = event_slice.iloc[0]['close']
            df.loc[event_mask, 'event_price'] = event_price
            
            # First Impulse Bar Stats
            event_slice['is_impulse_bar'] = detect_impulse_bar(event_slice)
            impulse_rows = event_slice[event_slice['is_impulse_bar']]
            if not impulse_rows.empty:
                first_impulse = impulse_rows.iloc[0]
                df.loc[event_mask, 'impulse_high'] = first_impulse['high']
                df.loc[event_mask, 'impulse_low'] = first_impulse['low']
                
            # First 5m Stats using get_time_window_mask
            five_min_mask = get_time_window_mask(event_slice, event_time, duration_seconds=300)
            five_min_slice = event_slice[five_min_mask]
            
            if not five_min_slice.empty:
                df.loc[event_mask, 'first_5m_high'] = five_min_slice['high'].max()
                df.loc[event_mask, 'first_5m_low'] = five_min_slice['low'].min()
                df.loc[event_mask, 'peak_1m_vol_5m'] = five_min_slice['vol'].max()
                
            # VWAP Uptime
            if not vwap_series.isna().all():
                above_vwap = event_slice['close'] > vwap_series
                uptime = above_vwap.mean()
                df.loc[event_mask, 'vwap_uptime'] = uptime

    # --- Targets ---
    print("Calculating targets...")
    
    # Forward Returns (Shorter intervals for earnings reactions)
    # 30s, 1m, 2m, 4m, 10m
    returns_df = get_forward_returns(df, horizons_seconds=[30, 60, 120, 240, 600])
    df = pd.concat([df, returns_df], axis=1)
    
    # Max Excursion (30m window - capped to event reaction period)
    excursion_df = get_max_excursion(df, window_seconds=1800, min_periods=60)
    df = pd.concat([df, excursion_df], axis=1)
    
    # Hit Target % (30m window)
    df['hit_1pct_30m'] = get_hit_target_pct(df, target_pct=0.01, window_seconds=1800)
    df['hit_2pct_30m'] = get_hit_target_pct(df, target_pct=0.02, window_seconds=1800)
    
    # First Touch Targets (TP/SL) - 30m window capped to event reaction
    df['target_1R_1pct'] = get_first_touch_target(df, tp_pct=0.01, sl_pct=0.01, window_seconds=1800)
    df['target_2R_1pct'] = get_first_touch_target(df, tp_pct=0.02, sl_pct=0.01, window_seconds=1800)

    # --- Filter Output to Event Windows Only ---
    print("Filtering to event windows...")
    if 'event_plus_minus_30m' in df.columns:
        df_filtered = df[df['event_plus_minus_30m'] == True].copy()
        print(f"Filtered from {len(df)} to {len(df_filtered)} rows.")
    else:
        print("Warning: 'event_plus_minus_30m' column not found. Outputting all rows.")
        df_filtered = df

    # Save output
    print(f"Saving to {output_file}...")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df_filtered.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add signals and targets to labeled OHLCV data.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output CSV file")
    parser.add_argument("--pre-labeled", action='store_true', default=True,
                        help="Input already has window labels (default: True)")
    
    args = parser.parse_args()
    
    add_earnings_signals(args.input_file, args.output_file, args.pre_labeled)
