import pandas as pd
import argparse
import os
from feature_engineering.trading_hours import label_trading_hours
from feature_engineering.windows import get_window_masks, get_n_days_before_mask, get_n_days_after_mask

def add_features(input_file, output_file, n_before=5, n_after=5):
    """
    Loads merged OHLCV data, adds feature columns, and saves to CSV.
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

    # 1. Add Trading Session Label
    print("Adding trading session labels...")
    df['trading_session'] = label_trading_hours(df['timestamp'])

    # 2. Add Window Masks
    # Initialize columns
    col_n_before = f'{n_before}_days_before'
    col_n_after = f'{n_after}_days_after'
    
    window_cols = [
        'event_plus_minus_30m', 
        'event_plus_30_to_120m', 
        'event_0_to_120m',
        'next_open_plus_minus_30m',
        col_n_before,
        col_n_after
    ]
    
    for col in window_cols:
        df[col] = False

    # Iterate over unique events to apply masks correctly
    unique_events = df['acceptance_datetime_utc'].unique()
    print(f"Found {len(unique_events)} unique events.")

    for event_time in unique_events:
        if pd.isna(event_time):
            continue
            
        # Generate masks for this event
        # Note: These functions return masks for the ENTIRE dataframe relative to this event_time
        fixed_masks = get_window_masks(df, event_time)
        mask_before = get_n_days_before_mask(df, event_time, n_days=n_before)
        mask_after = get_n_days_after_mask(df, event_time, n_days=n_after)
        
        # Identify rows belonging to this event
        event_rows = df['acceptance_datetime_utc'] == event_time
        
        # Apply masks only to the relevant rows
        # Logic: Row is in window IF (Row belongs to Event) AND (Row is in Window of Event)
        
        df.loc[event_rows, 'event_plus_minus_30m'] = fixed_masks['event_plus_minus_30m'][event_rows]
        df.loc[event_rows, 'event_plus_30_to_120m'] = fixed_masks['event_plus_30_to_120m'][event_rows]
        df.loc[event_rows, 'event_0_to_120m'] = fixed_masks['event_0_to_120m'][event_rows]
        df.loc[event_rows, 'next_open_plus_minus_30m'] = fixed_masks['next_open_plus_minus_30m'][event_rows]
        df.loc[event_rows, col_n_before] = mask_before[event_rows]
        df.loc[event_rows, col_n_after] = mask_after[event_rows]

    # Save output
    print(f"Saving to {output_file}...")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add features to merged OHLCV data.")
    parser.add_argument("input_file", help="Path to input merged CSV file")
    parser.add_argument("output_file", help="Path to output CSV file")
    parser.add_argument("--n_before", type=int, default=5, help="Number of days for n_days_before window")
    parser.add_argument("--n_after", type=int, default=5, help="Number of days for n_days_after window")
    
    args = parser.parse_args()
    
    add_features(args.input_file, args.output_file, args.n_before, args.n_after)
