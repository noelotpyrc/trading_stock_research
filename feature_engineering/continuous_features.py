"""
Continuous technical indicators with adaptive parameters.

This module provides technical indicators (RSI, EMA, ATR, etc.) with quality-based
adaptive parameters. HIGH quality tickers use 1s resampling, MEDIUM use 2s.
"""

import pandas as pd
import numpy as np

# Quality tier ticker lists (from earnings gap analysis)
HIGH_TICKERS = [
    'AAPL', 'ADBE', 'AMD', 'AVGO', 'CRM', 'C', 'GOOG', 'INTC',
    'META', 'MSFT', 'MU', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'QCOM',
    'SBUX', 'SNOW', 'UPST', 'WMT'
]

MEDIUM_TICKERS = [
    'BAC', 'COST', 'FDX', 'JPM', 'MDB', 'MS', 'NOW',
    'SHOP', 'SOFI', 'TGT', 'WFC'
]


def resample_ohlcv(df, freq='1s'):
    """
    Resample to regular frequency using pandas.
    
    Args:
        df: DataFrame with timestamp index or column
        freq: Resampling frequency ('1s', '2s', etc.)
    
    Returns:
        Resampled DataFrame with forward-filled values
    """
    # Ensure timestamp is index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # Resample OHLCV data
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'num': 'sum'
    })
    
    # Forward fill missing values
    resampled = resampled.ffill()
    
    # Reset index to get timestamp as column
    return resampled.reset_index()


def calculate_rsi(series, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        series: Price series (typically close)
        period: Lookback period
    
    Returns:
        RSI values (0-100)
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ema(series, period=14):
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        series: Price series
        period: Span for EMA
    
    Returns:
        EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with high, low, close
        period: Lookback period
    
    Returns:
        ATR values
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    return atr


def calculate_roc(series, period=14):
    """
    Calculate Rate of Change (ROC).
    
    Args:
        series: Price series
        period: Lookback period
    
    Returns:
        ROC as percentage
    """
    roc = ((series - series.shift(period)) / series.shift(period)) * 100
    return roc


def add_continuous_indicators(df, ticker_quality='HIGH'):
    """
    Add continuous technical indicators with adaptive parameters.
    
    IMPORTANT: This processes each event window separately to avoid creating
    millions of resampled bars across the entire time range.
    
    Args:
        df: DataFrame with timestamp, OHLCV columns, and event_plus_minus_30m flag
        ticker_quality: 'HIGH' or 'MEDIUM' (determines resampling freq and periods)
    
    Returns:
        DataFrame with added indicator columns
    """
    # Quality-specific parameters
    if ticker_quality == 'HIGH':
        resample_freq = '1s'
        rsi_period = 14
        ema_short = 8
        ema_mid = 21
        ema_long = 50
        atr_period = 14
        roc_period = 14
        vol_period = 20
    else:  # MEDIUM
        resample_freq = '2s'  
        rsi_period = 10
        ema_short = 5
        ema_mid = 15
        ema_long = 35
        atr_period = 10
        roc_period = 10
        vol_period = 15
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Process each event window separately to avoid memory explosion
    if 'acceptance_datetime_utc' in df.columns:
        print(f"  Processing {df['acceptance_datetime_utc'].nunique()} events separately...")
        
        # Group by event
        events = df['acceptance_datetime_utc'].unique()
        results = []
        
        for event_time in events:
            # Get rows for this event (±30 min window)
            event_mask = df['acceptance_datetime_utc'] == event_time
            event_df = df[event_mask].copy()
            
            if len(event_df) <  10:
                # Too few bars, skip indicator calculation
                # Add NaN columns
                for col in ['rsi', 'roc', 'ema_short', 'ema_mid', 'ema_long', 'atr', 'atr_pct', 'vol_ma', 'vol_ratio']:
                    event_df[col] = np.nan
                results.append(event_df)
                continue
            
            # Resample this event window only
            event_resampled = resample_ohlcv(
                event_df[['timestamp', 'open', 'high', 'low', 'close', 'vol', 'num']].copy(),
                resample_freq
            )
            
            # Calculate indicators
            event_resampled['rsi'] = calculate_rsi(event_resampled['close'], rsi_period)
            event_resampled['roc'] = calculate_roc(event_resampled['close'], roc_period)
            event_resampled['ema_short'] = calculate_ema(event_resampled['close'], ema_short)
            event_resampled['ema_mid'] = calculate_ema(event_resampled['close'], ema_mid)
            event_resampled['ema_long'] = calculate_ema(event_resampled['close'], ema_long)
            event_resampled['atr'] = calculate_atr(event_resampled, atr_period)
            event_resampled['atr_pct'] = (event_resampled['atr'] / event_resampled['close']) * 100
            event_resampled['vol_ma'] = event_resampled['vol'].rolling(vol_period, min_periods=1).mean()
            event_resampled['vol_ratio'] = event_resampled['vol'] / event_resampled['vol_ma']
            
            # Merge back using merge_asof
            indicator_cols = ['timestamp', 'rsi', 'roc', 'ema_short', 'ema_mid', 'ema_long',
                              'atr', 'atr_pct', 'vol_ma', 'vol_ratio']
            
            event_result = pd.merge_asof(
                event_df,
                event_resampled[indicator_cols],
                on='timestamp',
                direction='nearest'
            )
            
            results.append(event_result)
        
        # Combine all events
        final_df = pd.concat(results, ignore_index=True)
        print(f"  ✓ Added indicators ({len(final_df):,} rows)")
        return final_df
        
    else:
        # Fallback: process entire dataframe (not recommended for large datasets)
        print(f"  Warning: No event grouping, processing {len(df):,} rows as single block")
        df_resampled = resample_ohlcv(df[['timestamp', 'open', 'high', 'low', 'close', 'vol', 'num']].copy(),
                                       resample_freq)
        
        # Calculate indicators
        df_resampled['rsi'] = calculate_rsi(df_resampled['close'], rsi_period)
        df_resampled['roc'] = calculate_roc(df_resampled['close'], roc_period)
        df_resampled['ema_short'] = calculate_ema(df_resampled['close'], ema_short)
        df_resampled['ema_mid'] = calculate_ema(df_resampled['close'], ema_mid)
        df_resampled['ema_long'] = calculate_ema(df_resampled['close'], ema_long)
        df_resampled['atr'] = calculate_atr(df_resampled, atr_period)
        df_resampled['atr_pct'] = (df_resampled['atr'] / df_resampled['close']) * 100
        df_resampled['vol_ma'] = df_resampled['vol'].rolling(vol_period, min_periods=1).mean()
        df_resampled['vol_ratio'] = df_resampled['vol'] / df_resampled['vol_ma']
        
        # Merge back
        indicator_cols = ['timestamp', 'rsi', 'roc', 'ema_short', 'ema_mid', 'ema_long',
                          'atr', 'atr_pct', 'vol_ma', 'vol_ratio']
        
        result = pd.merge_asof(
            df,
            df_resampled[indicator_cols],
            on='timestamp',
            direction='nearest'
        )
        
        return result



def get_ticker_quality(ticker):
    """Get quality tier for a ticker."""
    if ticker in HIGH_TICKERS:
        return 'HIGH'
    elif ticker in MEDIUM_TICKERS:
        return 'MEDIUM'
    else:
        return 'UNKNOWN'
