import pandas as pd
import numpy as np

def get_opening_range(df, start_time, duration_seconds):
    """
    Calculates the Opening Range (High and Low) for a specified duration after start_time.
    
    Returns:
        tuple: (OR_high, OR_low) or (None, None) if no data in window.
    """
    end_time = start_time + pd.Timedelta(seconds=duration_seconds)
    
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    window_df = df[mask]
    
    if window_df.empty:
        return None, None
        
    or_high = window_df['high'].max()
    or_low = window_df['low'].min()
    
    return or_high, or_low

def detect_volume_spike(df, window=20, multiplier=3.0):
    """
    Detects volume spikes.
    
    Returns:
        Series (bool): True where volume > multiplier * rolling_mean(volume).
    """
    vol = df['vol']
    rolling_mean = vol.rolling(window=window).mean()
    
    # Handle NaN in rolling mean (start of file)
    is_spike = vol > (multiplier * rolling_mean)
    return is_spike.fillna(False)

def detect_impulse_bar(df, vol_mult=3.0, min_range=None):
    """
    Identifies impulse bars.
    
    Criteria:
    1. Volume Spike (vs 20-period MA by default)
    2. Large Range (optional min_range)
    
    Returns:
        Series (bool): True for impulse bars.
    """
    # 1. Volume Spike
    # Note: Strategy doc uses 20-sec MA for impulse vol check
    is_vol_spike = detect_volume_spike(df, window=20, multiplier=vol_mult)
    
    # 2. Range Check
    bar_range = df['high'] - df['low']
    
    if min_range is not None:
        is_large_range = bar_range >= min_range
        return is_vol_spike & is_large_range
    
    return is_vol_spike

def calculate_vwap_uptime(df, vwap_series, start_time=None):
    """
    Calculates the percentage of bars where Close > VWAP.
    
    Args:
        df: DataFrame with 'close'.
        vwap_series: Series containing VWAP values.
        start_time: Optional start time to filter the calculation.
        
    Returns:
        float: Percentage (0.0 to 1.0).
    """
    if start_time is not None:
        mask = df['timestamp'] >= start_time
        relevant_close = df.loc[mask, 'close']
        relevant_vwap = vwap_series.loc[mask]
    else:
        relevant_close = df['close']
        relevant_vwap = vwap_series
        
    if len(relevant_close) == 0:
        return 0.0
        
    above_count = (relevant_close > relevant_vwap).sum()
    return above_count / len(relevant_close)
