import pandas as pd
import numpy as np

# =============================================================================
# EARLY EVENT META FEATURES (0-5m)
# These features summarize price-volume dynamics in the first 5 minutes after
# an earnings event. They are designed to predict CVD-return correlation strength.
# =============================================================================

def calculate_delta_efficiency(event_slice_5m, event_price):
    """
    Delta Efficiency: How much price moved per unit of normalized net order flow.
    
    Formula: return_5m_pct / (cumsum(delta)_5m / total_vol_5m)
    
    Args:
        event_slice_5m: DataFrame with 0-5m data (must have 'close', 'delta', 'vol')
        event_price: Price at event time (t=0)
        
    Returns:
        float: Delta efficiency (can be positive or negative)
    """
    if event_slice_5m.empty or event_price <= 0:
        return np.nan
    
    close_5m = event_slice_5m['close'].iloc[-1]
    return_5m_pct = (close_5m - event_price) / event_price * 100
    
    net_delta = event_slice_5m['delta'].sum()
    total_vol = event_slice_5m['vol'].sum()
    
    if total_vol == 0:
        return np.nan
    
    normalized_delta = net_delta / total_vol  # -1 to +1 range
    
    if abs(normalized_delta) < 1e-6:
        return np.nan  # Avoid division by near-zero
    
    return return_5m_pct / normalized_delta


def calculate_cvd_monotonicity_sign(event_slice_5m):
    """
    CVD Monotonicity (Sign Method): How one-directional was the order flow?
    
    Formula: abs(mean(sign(delta))) in 0-5m
    
    Returns:
        float: 0-1 scale; 1.0 = every bar had same-sign delta (all buying or selling)
    """
    if event_slice_5m.empty or 'delta' not in event_slice_5m.columns:
        return np.nan
    
    delta = event_slice_5m['delta'].dropna()
    if len(delta) == 0:
        return np.nan
    
    signs = np.sign(delta)
    return abs(signs.mean())


def calculate_cvd_monotonicity_net_gross(event_slice_5m):
    """
    CVD Monotonicity (Net/Gross Method): What fraction of flow was directional vs canceled out?
    
    Formula: abs(cumsum(delta)) / sum(abs(delta)) in 0-5m
    
    Returns:
        float: 0-1 scale; 1.0 = all delta was in same direction (no cancellation)
    """
    if event_slice_5m.empty or 'delta' not in event_slice_5m.columns:
        return np.nan
    
    delta = event_slice_5m['delta'].dropna()
    if len(delta) == 0:
        return np.nan
    
    net_delta = abs(delta.sum())
    gross_delta = delta.abs().sum()
    
    if gross_delta == 0:
        return np.nan
    
    return net_delta / gross_delta


def calculate_price_cvd_ratio(first_5m_range_pct, cvd_zscore_max_2_5m, cvd_zscore_min_2_5m):
    """
    Price Range vs CVD Range: How much did price move relative to order flow extremes?
    
    Formula: first_5m_range_pct / abs(cvd_zscore_max_2_5m - cvd_zscore_min_2_5m)
    
    Uses 2-5m window for z-score to ensure meaningful values.
    
    Args:
        first_5m_range_pct: Price range in first 5m as % of event price
        cvd_zscore_max_2_5m: Max cvd_zscore in 2-5m window
        cvd_zscore_min_2_5m: Min cvd_zscore in 2-5m window
        
    Returns:
        float: Ratio (higher = price moved more than CVD suggested)
    """
    if pd.isna(first_5m_range_pct) or pd.isna(cvd_zscore_max_2_5m) or pd.isna(cvd_zscore_min_2_5m):
        return np.nan
    
    cvd_range = abs(cvd_zscore_max_2_5m - cvd_zscore_min_2_5m)
    
    if cvd_range < 1e-6:
        return np.nan  # Avoid division by near-zero
    
    return first_5m_range_pct / cvd_range


def calculate_vwap_deviation_5m(event_slice, seconds_col='seconds_since_event', 
                                  vwap_dist_col='vwap_distance_pct', target_sec=300):
    """
    VWAP Deviation at 5m: How far is event-anchored VWAP from event price at 5 minutes?
    
    Args:
        event_slice: DataFrame with event data
        seconds_col: Column name for seconds since event
        vwap_dist_col: Column name for VWAP distance %
        target_sec: Target time in seconds (default 300 = 5m)
        
    Returns:
        float: Absolute VWAP distance % at target time (or closest available)
    """
    if event_slice.empty:
        return np.nan
    
    if seconds_col not in event_slice.columns or vwap_dist_col not in event_slice.columns:
        return np.nan
    
    # Get row closest to target_sec
    closest_idx = (event_slice[seconds_col] - target_sec).abs().idxmin()
    vwap_dist = event_slice.loc[closest_idx, vwap_dist_col]
    
    if pd.isna(vwap_dist):
        return np.nan
    
    return abs(vwap_dist)


def calculate_vol_front_loading(event_slice_5m, seconds_col='seconds_since_event'):
    """
    Volume Front-Loading: Was volume concentrated in first half or second half of 0-5m?
    
    Formula: sum(vol[0-2.5m]) / sum(vol[2.5-5m])
    
    Returns:
        float: Ratio; >1 = front-loaded (exhaustion), <1 = back-loaded (building)
    """
    if event_slice_5m.empty or 'vol' not in event_slice_5m.columns:
        return np.nan
    
    if seconds_col not in event_slice_5m.columns:
        return np.nan
    
    first_half = event_slice_5m[event_slice_5m[seconds_col] <= 150]  # 0-2.5m
    second_half = event_slice_5m[(event_slice_5m[seconds_col] > 150) & 
                                  (event_slice_5m[seconds_col] <= 300)]  # 2.5-5m
    
    vol_first = first_half['vol'].sum()
    vol_second = second_half['vol'].sum()
    
    if vol_second == 0:
        return np.nan  # Avoid division by zero
    
    return vol_first / vol_second


def calculate_early_event_meta_features(event_slice, event_price, 
                                         cvd_zscore_max_2_5m=None, 
                                         cvd_zscore_min_2_5m=None,
                                         first_5m_range_pct=None):
    """
    Convenience function to calculate all early event meta features at once.
    
    Args:
        event_slice: Full event DataFrame (will be filtered to 0-5m internally)
        event_price: Price at event time
        cvd_zscore_max_2_5m: Pre-calculated max cvd_zscore in 2-5m (optional)
        cvd_zscore_min_2_5m: Pre-calculated min cvd_zscore in 2-5m (optional)
        first_5m_range_pct: Pre-calculated first 5m range % (optional)
        
    Returns:
        dict: Dictionary with all meta feature values
    """
    # Filter to 0-5m
    if 'seconds_since_event' in event_slice.columns:
        slice_5m = event_slice[event_slice['seconds_since_event'] <= 300].copy()
    else:
        slice_5m = event_slice.copy()
    
    # Calculate 2-5m cvd_zscore stats if not provided
    if cvd_zscore_max_2_5m is None or cvd_zscore_min_2_5m is None:
        if 'seconds_since_event' in event_slice.columns and 'cvd_zscore' in event_slice.columns:
            slice_2_5m = event_slice[(event_slice['seconds_since_event'] >= 120) & 
                                      (event_slice['seconds_since_event'] <= 300)]
            if not slice_2_5m.empty:
                cvd_zscore_max_2_5m = slice_2_5m['cvd_zscore'].max()
                cvd_zscore_min_2_5m = slice_2_5m['cvd_zscore'].min()
    
    # Calculate first_5m_range_pct if not provided
    if first_5m_range_pct is None:
        if not slice_5m.empty and event_price > 0:
            range_5m = slice_5m['high'].max() - slice_5m['low'].min()
            first_5m_range_pct = range_5m / event_price * 100
    
    return {
        'delta_efficiency_5m': calculate_delta_efficiency(slice_5m, event_price),
        'cvd_monotonicity_sign_5m': calculate_cvd_monotonicity_sign(slice_5m),
        'cvd_monotonicity_net_gross_5m': calculate_cvd_monotonicity_net_gross(slice_5m),
        'price_cvd_ratio_5m': calculate_price_cvd_ratio(first_5m_range_pct, 
                                                         cvd_zscore_max_2_5m, 
                                                         cvd_zscore_min_2_5m),
        'vwap_deviation_5m': calculate_vwap_deviation_5m(event_slice),
        'vol_front_loading_5m': calculate_vol_front_loading(slice_5m)
    }


# =============================================================================
# EXISTING MICROSTRUCTURE FUNCTIONS
# =============================================================================

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
