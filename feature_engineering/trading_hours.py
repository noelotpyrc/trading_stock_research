import pandas as pd
import pytz

# US Market Hours (ET)
PRE_MARKET_START = "04:00"
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
POST_MARKET_END = "20:00"

def label_trading_hours(timestamps):
    """
    Labels timestamps as 'Pre', 'Regular', 'Post', or 'Overnight' based on US Equity hours.
    
    Args:
        timestamps (pd.Series): Series of timestamps (UTC, nanoseconds or datetime objects).
        
    Returns:
        pd.Series: Series of strings labeling the session.
    """
    # Convert to datetime if needed and ensure UTC
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        ts_utc = pd.to_datetime(timestamps, utc=True)
    else:
        ts_utc = timestamps.dt.tz_convert('UTC') if timestamps.dt.tz is not None else timestamps.dt.tz_localize('UTC')

    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    ts_et = ts_utc.dt.tz_convert(eastern)
    
    # Extract time component for comparison
    times = ts_et.dt.strftime('%H:%M')
    
    # Initialize with 'Overnight'
    labels = pd.Series('Overnight', index=timestamps.index)
    
    # Create masks
    is_pre = (times >= PRE_MARKET_START) & (times < MARKET_OPEN)
    is_reg = (times >= MARKET_OPEN) & (times < MARKET_CLOSE)
    is_post = (times >= MARKET_CLOSE) & (times < POST_MARKET_END)
    
    # Apply labels
    labels[is_pre] = 'Pre'
    labels[is_reg] = 'Regular'
    labels[is_post] = 'Post'
    
    # Handle weekends? For now, assuming data is only on trading days or logic applies to time only.
    # If needed, we can add dayofweek check.
    
    return labels
