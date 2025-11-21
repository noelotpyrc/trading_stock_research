import pandas as pd

def get_seconds_since_event(df, event_time):
    """
    Calculates seconds elapsed since the event time for each row.
    
    Returns:
        Series (float): Seconds since event. Negative for pre-event.
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        # Try to convert if not, but ideally should be handled upstream
        timestamps = pd.to_datetime(df['timestamp'], utc=True)
    else:
        timestamps = df['timestamp']
        
    delta = timestamps - event_time
    return delta.dt.total_seconds()

def filter_window(df, start_time, duration_seconds):
    """
    Returns a slice of the DataFrame starting at start_time for duration_seconds.
    """
    end_time = start_time + pd.Timedelta(seconds=duration_seconds)
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    return df[mask].copy()
