import pandas as pd
import pytz
from datetime import timedelta
from .trading_hours import label_trading_hours, MARKET_OPEN, MARKET_CLOSE, PRE_MARKET_START, POST_MARKET_END

def get_next_open(event_time_et):
    """
    Determines the next regular market open time relative to the event.
    """
    event_date = event_time_et.date()
    event_time_str = event_time_et.strftime('%H:%M')
    
    # If before market open (e.g. 8:00), next open is today 9:30
    if event_time_str < MARKET_OPEN:
        next_open = pd.Timestamp(event_date).tz_localize('US/Eastern') + pd.Timedelta(hours=9, minutes=30)
    else:
        # If during or after market, next open is tomorrow 9:30
        # Note: This assumes next day is a trading day. 
        # For robust production use, would need a trading calendar.
        # For now, simple +1 day logic as requested/implied.
        next_open = pd.Timestamp(event_date + timedelta(days=1)).tz_localize('US/Eastern') + pd.Timedelta(hours=9, minutes=30)
        
    return next_open

def get_window_masks(df, event_time_utc):
    """
    Generates boolean masks for the 3 fixed time windows (Event +/- 30m, Event + 30-240m, Next Open +/- 30m).
    
    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' column (nanoseconds or datetime).
        event_time_utc (str or pd.Timestamp): Earnings release time in UTC.
        
    Returns:
        dict: Dictionary of boolean Series masks.
    """
    # Ensure DF timestamps are datetime UTC
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        ts_utc = pd.to_datetime(df['timestamp'], utc=True)
    else:
        ts_utc = df['timestamp'].dt.tz_convert('UTC') if df['timestamp'].dt.tz is not None else df['timestamp'].dt.tz_localize('UTC')

    # Parse event time
    event_ts = pd.to_datetime(event_time_utc, utc=True)
    
    # Convert to ET for logic
    eastern = pytz.timezone('US/Eastern')
    event_et = event_ts.tz_convert(eastern)
    
    masks = {}
    
    # --- Window 2: Event +/- 30 mins ---
    w2_start = event_ts - pd.Timedelta(minutes=30)
    w2_end = event_ts + pd.Timedelta(minutes=30)
    masks['event_plus_minus_30m'] = (ts_utc >= w2_start) & (ts_utc <= w2_end)
    
    # --- Window 3: Event + 30 to + 120 mins ---
    w3_start = event_ts + pd.Timedelta(minutes=30)
    w3_end_nominal = event_ts + pd.Timedelta(minutes=120)
    
    current_day_end = pd.Timestamp(event_et.date()).tz_localize('US/Eastern') + pd.Timedelta(hours=20)
    w3_cutoff = current_day_end.tz_convert('UTC')
    
    w3_end = min(w3_end_nominal, w3_cutoff)
    
    masks['event_plus_30_to_120m'] = (ts_utc > w3_start) & (ts_utc <= w3_end)
    
    # --- Window: Event 0 to +120 mins (post-event extended) ---
    w_ext_start = event_ts
    w_ext_end_nominal = event_ts + pd.Timedelta(minutes=120)
    w_ext_end = min(w_ext_end_nominal, w3_cutoff)
    
    masks['event_0_to_120m'] = (ts_utc >= w_ext_start) & (ts_utc <= w_ext_end)
    
    # --- Window 4: Next Open +/- 30 mins ---
    next_open_et = get_next_open(event_et)
    next_open_utc = next_open_et.tz_convert('UTC')
    
    w4_start = next_open_utc - pd.Timedelta(minutes=30)
    w4_end = next_open_utc + pd.Timedelta(minutes=30)
    masks['next_open_plus_minus_30m'] = (ts_utc >= w4_start) & (ts_utc <= w4_end)
    
    return masks

def get_n_days_before_mask(df, event_time_utc, n_days=5):
    """Generates mask for N trading days before event."""
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        ts_utc = pd.to_datetime(df['timestamp'], utc=True)
    else:
        ts_utc = df['timestamp'].dt.tz_convert('UTC') if df['timestamp'].dt.tz is not None else df['timestamp'].dt.tz_localize('UTC')

    event_ts = pd.to_datetime(event_time_utc, utc=True)
    eastern = pytz.timezone('US/Eastern')
    event_et = event_ts.tz_convert(eastern)
    event_time_str = event_et.strftime('%H:%M')
    
    if event_time_str < MARKET_CLOSE:
        w1_end_et = pd.Timestamp(event_et.date() - timedelta(days=1)).tz_localize('US/Eastern') + pd.Timedelta(hours=20)
    else:
        w1_end_et = pd.Timestamp(event_et.date()).tz_localize('US/Eastern') + pd.Timedelta(hours=16)
        
    w1_end_utc = w1_end_et.tz_convert('UTC')
    w1_start_utc = w1_end_utc - pd.Timedelta(days=n_days)
    
    return (ts_utc >= w1_start_utc) & (ts_utc < w1_end_utc)

def get_n_days_after_mask(df, event_time_utc, n_days=5):
    """Generates mask for N trading days after event."""
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        ts_utc = pd.to_datetime(df['timestamp'], utc=True)
    else:
        ts_utc = df['timestamp'].dt.tz_convert('UTC') if df['timestamp'].dt.tz is not None else df['timestamp'].dt.tz_localize('UTC')

    event_ts = pd.to_datetime(event_time_utc, utc=True)
    eastern = pytz.timezone('US/Eastern')
    event_et = event_ts.tz_convert(eastern)
    event_time_str = event_et.strftime('%H:%M')
    
    if event_time_str < MARKET_CLOSE:
        w5_start_et = pd.Timestamp(event_et.date() + timedelta(days=1)).tz_localize('US/Eastern') + pd.Timedelta(hours=4)
    else:
        w5_start_et = pd.Timestamp(event_et.date() + timedelta(days=1)).tz_localize('US/Eastern') + pd.Timedelta(hours=9, minutes=30)
        
    w5_start_utc = w5_start_et.tz_convert('UTC')
    w5_end_utc = w5_start_utc + pd.Timedelta(days=n_days)
    
    return (ts_utc >= w5_start_utc) & (ts_utc <= w5_end_utc)

def get_time_window_mask(df, start_time, duration_seconds):
    """
    Generates boolean mask for rows within a time window.
    
    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' column.
        start_time (str or pd.Timestamp): Start time in UTC.
        duration_seconds (int): Duration in seconds.
        
    Returns:
        pd.Series: Boolean mask for rows in [start_time, start_time + duration].
    """
    # Ensure DF timestamps are datetime UTC
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        ts_utc = pd.to_datetime(df['timestamp'], utc=True)
    else:
        ts_utc = df['timestamp'].dt.tz_convert('UTC') if df['timestamp'].dt.tz is not None else df['timestamp'].dt.tz_localize('UTC')

    start_ts = pd.to_datetime(start_time, utc=True)
    end_ts = start_ts + pd.Timedelta(seconds=duration_seconds)
    
    return (ts_utc >= start_ts) & (ts_utc <= end_ts)
