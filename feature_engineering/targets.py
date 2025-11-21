import pandas as pd
import numpy as np

def get_forward_returns(df, horizons_seconds=[300, 900, 1800], tolerance_seconds=30):
    """
    Calculates forward returns for specified time horizons in seconds.
    
    Args:
        df: DataFrame with 'timestamp' and 'close' columns.
        horizons_seconds: List of horizons in seconds (default: 5m, 15m, 30m).
        tolerance_seconds: Maximum time difference to match (default: 30s).
        
    Returns:
        DataFrame with target columns (e.g., 'target_ret_300s').
    """
    targets = pd.DataFrame(index=df.index)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
    # Sort by timestamp to ensure correct shifting (though input should be sorted)
    # We'll assume input is sorted for performance, or we can sort locally if needed.
    # For now, assuming sorted.
    
    # Since data might not be perfectly regular (e.g. missing minutes), 
    # we can't just shift by row count. We need to find the price N seconds later.
    # Using merge_asof is robust for this.
    
    temp_df = df[['timestamp', 'close']].copy()
    temp_df = temp_df.sort_values('timestamp')
    
    for seconds in horizons_seconds:
        target_col = f'target_ret_{seconds}s'
        
        # Create a lookup dataframe with shifted timestamps
        lookup_df = temp_df.copy()
        lookup_df['lookup_time'] = lookup_df['timestamp'] - pd.Timedelta(seconds=seconds)
        
        # Merge to find the close price 'seconds' ago (which is the 'current' row for the target)
        # Wait, simpler: For row T, we want Close at T+seconds.
        # So we take the dataframe, shift timestamp back by 'seconds', and merge that back to original.
        
        future_df = temp_df.copy()
        future_df['join_time'] = future_df['timestamp'] - pd.Timedelta(seconds=seconds)
        
        # We want to attach future_df['close'] to df where df['timestamp'] ~= future_df['join_time']
        # Actually, merge_asof direction='backward' on the *original* df looking for *future* time?
        
        # Let's do: For each row in df (time T), find row in df with time T + seconds.
        # target_time = T + seconds.
        # We want close at target_time.
        
        target_times = temp_df[['timestamp']].copy()
        target_times['target_time'] = target_times['timestamp'] + pd.Timedelta(seconds=seconds)
        
        merged = pd.merge_asof(
            target_times, 
            temp_df, 
            left_on='target_time', 
            right_on='timestamp', 
            direction='backward',
            tolerance=pd.Timedelta(seconds=tolerance_seconds)
        )
        
        # merged['close'] is the future close
        targets[target_col] = (merged['close'].values - df['close'].values) / df['close'].values
        
    return targets

def get_max_excursion(df, window_seconds=3600, min_periods=60):
    """
    Calculates Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE) 
    for a theoretical LONG position over the next window_seconds.
    
    Args:
        df: DataFrame with timestamp, high, low, close columns.
        window_seconds: Forward-looking window in seconds.
        min_periods: Minimum number of bars required (default: 60 for 1 min of second-bar data).
    
    Returns:
        DataFrame with 'mfe_{window_seconds}s' and 'mae_{window_seconds}s'.
    """
    # This is computationally expensive with pure pandas for large DFs if iterating.
    # Using rolling() on reversed index with time offset is one way, but pandas rolling time 
    # requires datetime index.
    
    temp_df = df[['timestamp', 'high', 'low', 'close']].copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df['timestamp']):
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], utc=True)
        
    temp_df = temp_df.set_index('timestamp').sort_index()
    
    # We want forward looking max/min. 
    # Reverse, rolling max/min, reverse back.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_seconds) 
    # Note: FixedForwardWindowIndexer takes integer window_size (rows), not time.
    # If data is 1-second bars, window_size=window_seconds.
    # If data is irregular, this is wrong.
    
    # Robust approach for irregular time series:
    # 1. Reindex to 1-second frequency (forward fill) -> huge memory?
    # 2. Use merge_asof or iterative approach?
    # 3. Use rolling with time offset on reversed dataframe?
    
    # Let's try the rolling time offset on reversed DF.
    # Rolling with time offset looks *backward*. 
    # If we reverse time, "backward" becomes "forward" in original time.
    
    # Reverse DF
    rev_df = temp_df.iloc[::-1]
    
    # Rolling max/min with time window
    # We want max high in [t, t + window]
    # In reversed, this is [t, t - window] (which is "past" in reversed time)
    # But rolling includes current point.
    
    # Note: rolling('1H') is closed='right' by default.
    
    rolling_max = rev_df['high'].rolling(f'{window_seconds}s', min_periods=min_periods).max()
    rolling_min = rev_df['low'].rolling(f'{window_seconds}s', min_periods=min_periods).min()
    
    # Re-reverse to match original order
    future_max = rolling_max.iloc[::-1].values
    future_min = rolling_min.iloc[::-1].values
    
    # Calculate MFE/MAE
    # MFE = (Max High - Entry) / Entry
    # MAE = (Min Low - Entry) / Entry
    
    mfe = (future_max - df['close'].values) / df['close'].values
    mae = (future_min - df['close'].values) / df['close'].values
    
    res = pd.DataFrame(index=df.index)
    res[f'mfe_{window_seconds}s'] = mfe
    res[f'mae_{window_seconds}s'] = mae
    
    return res

def get_hit_target_pct(df, target_pct, window_seconds):
    """
    Returns boolean: Did price hit +target_pct OR -target_pct within window_seconds?
    Useful for "Did it move X%?" check.
    """
    # Reuse MFE/MAE logic if available, or recompute.
    # If we already have MFE/MAE for this window, we can just check those.
    # For efficiency, let's assume we might call this independently.
    
    # To avoid recomputing rolling max/min every time, we might want to compute MFE/MAE once 
    # and derive this. But for now, implementing standalone.
    
    excursions = get_max_excursion(df, window_seconds)
    mfe_col = f'mfe_{window_seconds}s'
    mae_col = f'mae_{window_seconds}s'
    
    hit_upper = excursions[mfe_col] >= target_pct
    hit_lower = excursions[mae_col] <= -target_pct
    
    return hit_upper | hit_lower

def get_first_touch_target(df, tp_pct, sl_pct, window_seconds):
    """
    Simulates a Long position.
    Returns:
        1: Hit TP first
        -1: Hit SL first
        0: Hit neither within window
    
    tp_pct: positive float (e.g. 0.02 for +2%)
    sl_pct: positive float (e.g. 0.01 for -1%) -> we check if return <= -0.01
    """
    # This requires knowing *when* the high/low happened.
    # The simple rolling max/min doesn't give us the timestamp of the max/min.
    
    # We need to scan forward. For large datasets, vectorization is hard for "first touch".
    # However, we can approximate or use numba if performance is critical.
    # For this scale, let's try a reasonably efficient pandas approach.
    
    # We can check if High >= Entry * (1+TP) AND Low <= Entry * (1-SL) in the same bar?
    # If both happen in same bar/window, it's ambiguous without lower timeframe data.
    # We will assume:
    # If High hits TP and Low doesn't hit SL -> TP
    # If Low hits SL and High doesn't hit TP -> SL
    # If Both hit -> Ambiguous (return 0 or prioritize one? Conservative: SL hit first?)
    
    # But we need to know which happened *first* in the window.
    # This is hard with just rolling max/min.
    
    # Simplified approach for now:
    # 1. Filter rows where MFE >= TP OR MAE <= -SL.
    # 2. For those rows, we need to check the timing.
    
    # Given the complexity of "first touch" in vectorized pandas without tick data,
    # we will implement a simplified version:
    # If (MFE >= TP) and (MAE > -SL) -> TP (Hit TP, never hit SL)
    # If (MAE <= -SL) and (MFE < TP) -> SL (Hit SL, never hit TP)
    # If Both -> Ambiguous (Conflict). We can label as 'Conflict' or 0.
    
    excursions = get_max_excursion(df, window_seconds)
    mfe = excursions[f'mfe_{window_seconds}s']
    mae = excursions[f'mae_{window_seconds}s']
    
    tp_hit = mfe >= tp_pct
    sl_hit = mae <= -sl_pct
    
    result = pd.Series(0, index=df.index)
    
    # Clear TP wins
    result[tp_hit & (~sl_hit)] = 1
    
    # Clear SL wins
    result[sl_hit & (~tp_hit)] = -1
    
    # Conflicts: Both hit within window. We don't know which was first without finer granularity.
    # We will leave them as 0 (or could define a policy like 'SL always triggers first').
    # For now, 0 implies "Uncertain/Both".
    
    return result
