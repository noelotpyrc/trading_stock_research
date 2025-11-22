import pandas as pd
import numpy as np
import os
from feature_engineering.label_earning_event_windows import add_features as label_windows
from feature_engineering.add_earnings_signals_meta import add_earnings_signals

def create_dummy_data(filename):
    # Create 100 minutes of data
    dates = pd.date_range(start='2023-01-01 09:30:00', periods=100, freq='1min', tz='UTC')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'vol': np.random.randint(100, 1000, 100),
        'acceptance_datetime_utc': [dates[50]] * 100 # Event at index 50
    })
    
    # Fix high/low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    df.to_csv(filename, index=False)
    print(f"Created {filename}")

def test_pipeline():
    input_file = 'dummy_input.csv'
    labeled_file = 'dummy_labeled.csv'
    final_file = 'dummy_final.csv'
    
    create_dummy_data(input_file)
    
    print("\n--- Step 1: Label Windows ---")
    label_windows(input_file, labeled_file)
    
    print("\n--- Step 2: Add Signals (Filtered Output) ---")
    add_earnings_signals(labeled_file, final_file, pre_labeled=True)
    
    if os.path.exists(final_file):
        print("\n--- Verification ---")
        df_out = pd.read_csv(final_file)
        print(f"Output rows: {len(df_out)}")
        print("Columns:", df_out.columns.tolist())
        
        # Check for new columns (including normalized)
        expected_cols = [
            'cvd_since_event', 'vwap_since_event', 'or_high', 'or_low',
            'or_width_pct', 'cvd_pct_volume', 'vwap_distance_pct',
            'target_ret_30s', 'target_ret_60s', 'mfe_1800s', 'hit_1pct_30m', 'target_1R_1pct'
        ]
        missing = [c for c in expected_cols if c not in df_out.columns]
        
        if not missing:
            print("SUCCESS: All expected columns found.")
            
            # Verify filtering worked (should be ~60 rows for +/- 30m window, not 100)
            if len(df_out) < 100:
                print(f"✓ Output correctly filtered ({len(df_out)} rows vs 100 input rows)")
            else:
                print(f"✗ Filtering may have failed (output same size as input)")
            
            # Check signal logic
            valid_cvd = df_out['cvd_since_event'].notna().sum()
            print(f"Valid CVD count: {valid_cvd}")
            
            valid_or = df_out['or_high'].notna().sum()
            print(f"Valid OR count: {valid_or}")
            
            # Check TP/SL target values
            if 'target_1R_1pct' in df_out.columns:
                tp_sl_counts = df_out['target_1R_1pct'].value_counts()
                print(f"TP/SL Target Distribution:\n{tp_sl_counts}")
            
        else:
            print(f"FAILURE: Missing columns: {missing}")
    else:
        print("FAILURE: Output file not created.")

    # Cleanup
    for f in [input_file, labeled_file, final_file]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    test_pipeline()
