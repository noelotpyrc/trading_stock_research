import pandas as pd
import numpy as np

def estimate_delta(df):
    """
    Estimates volume delta based on the position of the close within the high-low range.
    
    Formula:
    delta = volume * clamp((close - midpoint) / half_range, -1, 1)
    
    Where:
    midpoint = (high + low) / 2
    half_range = (high - low) / 2
    
    If high == low, delta is 0.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['vol'] # Using 'vol' as per standard in this project, check if 'volume' exists if 'vol' missing
    
    # Handle case where high == low to avoid division by zero
    range_span = high - low
    midpoint = (high + low) / 2
    half_range = range_span / 2
    
    # Calculate position (-1 to 1)
    # We use np.where to handle the zero division case safely
    position = np.where(
        range_span > 0,
        (close - midpoint) / half_range,
        0.0
    )
    
    # Clamp position to [-1, 1] just in case, though math suggests it should be within range
    # (close can't be outside high/low)
    position = np.clip(position, -1.0, 1.0)
    
    delta = volume * position
    return pd.Series(delta, index=df.index, name='delta')

def calculate_cvd(df, start_time=None):
    """
    Calculates Cumulative Volume Delta (CVD).
    
    Args:
        df: DataFrame containing 'delta' column (or capable of calculating it).
        start_time: Optional timestamp to anchor the accumulation. 
                    If None, accumulates from start of DataFrame.
    """
    if 'delta' not in df.columns:
        delta = estimate_delta(df)
    else:
        delta = df['delta']
        
    if start_time is not None:
        # Filter for rows after start_time
        mask = df['timestamp'] >= start_time
        # We only sum the delta for the relevant period
        # But we want the Series to align with the original DF? 
        # Usually CVD is relevant only within the context of the session.
        # Let's return CVD for the whole DF, but reset to 0 at start_time?
        # Or just return the slice?
        # Strategy doc implies "cvd since event".
        
        # Let's return a Series aligned with df, but values before start_time are NaN or 0?
        # Better: Return values for the whole DF, but calculation starts at start_time.
        # Rows before start_time will be NaN.
        
        relevant_delta = delta.copy()
        relevant_delta[df['timestamp'] < start_time] = 0 # Or just ignore them
        
        # Actually, if we want "CVD since event", we should probably just slice it?
        # But keeping index alignment is good.
        
        # Let's do this: Set delta to 0 before start_time, then cumsum.
        # But that means CVD is 0 before start_time.
        
        # Alternative: Just slice.
        # Let's stick to returning a Series aligned with the input DF index.
        
        # Create a mask for valid times
        valid_mask = df['timestamp'] >= start_time
        cvd = pd.Series(np.nan, index=df.index, name='cvd')
        cvd[valid_mask] = delta[valid_mask].cumsum()
        return cvd
        
    return delta.cumsum().rename('cvd')

def calculate_anchored_vwap(df, start_time=None):
    """
    Calculates Anchored VWAP.
    
    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    """
    # Ensure we have price and volume
    # Typical Price is usually (H+L+C)/3, but strategy doc often implies just using Close or standard VWAP.
    # Standard VWAP uses Typical Price.
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    volume = df['vol']
    
    pv = typical_price * volume
    
    if start_time is not None:
        valid_mask = df['timestamp'] >= start_time
        
        cum_pv = pd.Series(np.nan, index=df.index)
        cum_vol = pd.Series(np.nan, index=df.index)
        
        cum_pv[valid_mask] = pv[valid_mask].cumsum()
        cum_vol[valid_mask] = volume[valid_mask].cumsum()
        
        return (cum_pv / cum_vol).rename('vwap')
    
    return (pv.cumsum() / volume.cumsum()).rename('vwap')

def calculate_rsi(series, period=14):
    """
    Calculates Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f'rsi_{period}')

def calculate_ema(series, period):
    """
    Calculates Exponential Moving Average (EMA).
    """
    return series.ewm(span=period, adjust=False).mean().rename(f'ema_{period}')

def calculate_rolling_zscore(series, window):
    """
    Calculates Rolling Z-Score.
    Z = (Value - Mean) / StdDev
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    zscore = (series - rolling_mean) / rolling_std
    return zscore.rename(f'zscore_{window}')
