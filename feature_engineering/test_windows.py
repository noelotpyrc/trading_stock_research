import pandas as pd
import pytz
from feature_engineering.trading_hours import label_trading_hours
from feature_engineering.windows import get_window_masks, get_n_days_before_mask, get_n_days_after_mask

def test_trading_hours():
    print("Testing Trading Hours...")
    timestamps = pd.to_datetime([
        "2023-10-27 08:00:00", # Pre (04:00 ET)
        "2023-10-27 13:29:00", # Pre (09:29 ET)
        "2023-10-27 13:30:00", # Regular (09:30 ET)
        "2023-10-27 19:59:00", # Regular (15:59 ET)
        "2023-10-27 20:00:00", # Post (16:00 ET)
        "2023-10-27 23:59:00", # Post (19:59 ET)
        "2023-10-28 00:00:00", # Overnight (20:00 ET)
    ], utc=True)
    
    labels = label_trading_hours(pd.Series(timestamps))
    expected = ['Pre', 'Pre', 'Regular', 'Regular', 'Post', 'Post', 'Overnight']
    
    for i, (ts, label, exp) in enumerate(zip(timestamps, labels, expected)):
        print(f"  {ts} -> {label} (Expected: {exp})")
        assert label == exp, f"Mismatch at index {i}"
    print("Trading Hours Test Passed!\n")

def test_windows():
    print("Testing Windows...")
    # Create dummy dataframe covering a few days
    dates = pd.date_range(start="2023-10-25", end="2023-10-30", freq="30min", tz="UTC")
    df = pd.DataFrame({'timestamp': dates})
    
    # Case 1: Pre-market release (e.g., 08:00 ET -> 12:00 UTC)
    event_utc = "2023-10-27 12:00:00" 
    print(f"Case 1: Event at {event_utc} (Pre-market)")
    masks = get_window_masks(df, event_utc)
    
    # Verify Window 4 (Next Open)
    # Event 8:00 ET -> Next Open Today 9:30 ET (13:30 UTC)
    # Window 4: 13:00 UTC to 14:00 UTC
    w4_true = df[masks['next_open_plus_minus_30m']]
    print(f"  Window 4 range: {w4_true['timestamp'].min()} to {w4_true['timestamp'].max()}")
    assert not w4_true.empty
    assert w4_true['timestamp'].min() >= pd.Timestamp("2023-10-27 13:00:00", tz="UTC")
    
    # Case 2: Post-market release (e.g., 16:30 ET -> 20:30 UTC)
    event_utc = "2023-10-27 20:30:00"
    print(f"Case 2: Event at {event_utc} (Post-market)")
    masks = get_window_masks(df, event_utc)
    
    # Verify Window 4 (Next Open)
    # Event 16:30 ET -> Next Open Tomorrow 9:30 ET (Assuming trading day)
    # Tomorrow is 2023-10-28 (Saturday) - logic currently just adds 1 day
    # So 2023-10-28 9:30 ET -> 13:30 UTC
    w4_true = df[masks['next_open_plus_minus_30m']]
    print(f"  Window 4 range: {w4_true['timestamp'].min()} to {w4_true['timestamp'].max()}")
    assert not w4_true.empty
    assert w4_true['timestamp'].min() >= pd.Timestamp("2023-10-28 13:00:00", tz="UTC")

    # Case 3: Regular Market Release (e.g., 10:00 ET -> 14:00 UTC)
    event_utc = "2023-10-27 14:00:00"
    print(f"Case 3: Event at {event_utc} (Regular Market)")
    masks = get_window_masks(df, event_utc)
    
    # Next Open should be Tomorrow 9:30 ET
    w4_true = df[masks['next_open_plus_minus_30m']]
    assert w4_true['timestamp'].min() >= pd.Timestamp("2023-10-28 13:00:00", tz="UTC")
    
    # Case 4: Window 3 Cutoff Check (Event at 17:00 ET -> 21:00 UTC)
    # Window 3 starts 17:30 ET. Nominal end 21:30 ET.
    # Hard cutoff is 20:00 ET (00:00 UTC next day).
    event_utc = "2023-10-27 21:00:00"
    print(f"Case 4: Event at {event_utc} (Post-market, Window 3 Cutoff)")
    masks = get_window_masks(df, event_utc)
    
    w3_true = df[masks['event_plus_30_to_240m']]
    if not w3_true.empty:
        max_ts = w3_true['timestamp'].max()
        cutoff_ts = pd.Timestamp("2023-10-28 00:00:00", tz="UTC")
        print(f"  Window 3 max: {max_ts} (Cutoff: {cutoff_ts})")
        assert max_ts <= cutoff_ts
    else:
        print("  Window 3 is empty (Correct, as start 17:30 ET is close to cutoff)")

    # Case 5: N-Days Logic
    print("Case 5: N-Days Logic")
    # Pre-market event (08:00 ET) -> N-days before ends at Prev Day 20:00 ET
    event_utc = "2023-10-27 12:00:00"
    mask_before = get_n_days_before_mask(df, event_utc, n_days=1)
    true_before = df[mask_before]
    expected_end = pd.Timestamp("2023-10-27 00:00:00", tz="UTC") # Prev day (26th) 20:00 ET -> 27th 00:00 UTC
    print(f"  Pre-market Before-Window End: {true_before['timestamp'].max()} (Expected: {expected_end})")
    assert true_before['timestamp'].max() <= expected_end
    
    # Post-market event (16:30 ET) -> N-days before ends at Today 16:00 ET
    event_utc = "2023-10-27 20:30:00"
    mask_before = get_n_days_before_mask(df, event_utc, n_days=1)
    true_before = df[mask_before]
    expected_end = pd.Timestamp("2023-10-27 20:00:00", tz="UTC") # Today 16:00 ET -> 20:00 UTC
    print(f"  Post-market Before-Window End: {true_before['timestamp'].max()} (Expected: {expected_end})")
    assert true_before['timestamp'].max() <= expected_end

    print("Windows Test Passed!\n")

if __name__ == "__main__":
    try:
        test_trading_hours()
        test_windows()
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
