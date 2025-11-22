#!/usr/bin/env python3
"""
Calculate baseline statistics from PRE-EARNING periods for each ticker.

Option B: Uses ORIGINAL merged second bar files, filters to pre-earning periods,
calculates indicators on that data, then computes baseline stats.

Process:
1. Load original merged file (full history)
2. Filter to 5_days_before == True (pre-earning periods)
3. Calculate RSI, ATR, VWAP on pre-earning data
4. Compute baseline statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering.continuous_features import (
    calculate_rsi, calculate_atr, calculate_ema, get_ticker_quality
)

# Paths
DATA_DIR = Path('/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv')
OUTPUT_FILE = Path('data/baseline_stats/ticker_baseline_stats.csv')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Quality tickers
QUALITY_TICKERS = [
    # HIGH
    'AAPL', 'ADBE', 'AMD', 'AVGO', 'CRM', 'C', 'GOOG', 'INTC',
    'META', 'MSFT', 'MU', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'QCOM',
    'SBUX', 'SNOW', 'UPST', 'WMT',
    # MEDIUM
    'BAC', 'COST', 'FDX', 'JPM', 'MDB', 'MS', 'NOW',
    'SHOP', 'SOFI', 'TGT', 'WFC'
]


def calculate_indicators_on_data(df):
    """Calculate RSI, ATR on the given data."""
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # ATR and ATR%
    df['atr'] = calculate_atr(df, period=14)
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Calculate VWAP (simple cumulative)
    df['cum_vol_price'] = (df['vw'] * df['vol']).cumsum()
    df['cum_vol'] = df['vol'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df['vwap_distance_pct'] = ((df['close'] - df['vwap']) / df['close']) * 100
    
    return df


def calculate_ticker_baseline(ticker_file):
    """
    Calculate baseline statistics from PRE-EARNING periods.
    
    Pre-earning period defined as:
    - 5 days before earnings event
    - Up until 30 minutes before earnings event (buffer to avoid pre-announcement volatility)
    
    Args:
        ticker_file: Path to original merged second bar file
    
    Returns:
        dict: Baseline statistics
    """
    ticker = ticker_file.stem.replace('_second_merged', '')
    
    print(f"  {ticker}...", end=" ", flush=True)
    
    # Load file
    df = pd.read_csv(ticker_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')
    
    # Check for earnings datetime column
    if 'acceptance_datetime_utc' not in df.columns:
        print("❌ No acceptance_datetime_utc column")
        return None
    
    # Get unique earnings events
    df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)
    earnings_times = df['acceptance_datetime_utc'].dropna().unique()
    
    if len(earnings_times) == 0:
        print("❌ No earnings events found")
        return None
    
    print(f"({len(earnings_times)} events)", end=" ", flush=True)
    
    # Identify pre-earning periods for all events
    # Pre-earning: [event - 5 days, event - 30 minutes]
    pre_earning_masks = []
    
    for event_time in earnings_times:
        # 5 days before the event
        start_time = event_time - pd.Timedelta(days=5)
        # 30 minutes before the event (buffer)
        end_time = event_time - pd.Timedelta(minutes=30)
        
        # Mask for this pre-earning period
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] < end_time)
        pre_earning_masks.append(mask)
    
    # Combine all pre-earning periods
    combined_mask = pd.concat(pre_earning_masks, axis=1).any(axis=1)
    pre_earning = df[combined_mask].copy()
    
    if len(pre_earning) < 1000:  # Need sufficient data for reliable stats
        print(f"⚠️  Insufficient data ({len(pre_earning)} rows)")
        return None
    
    print(f"({len(pre_earning):,} pre-earning rows)", end=" ", flush=True)
    
    # Calculate indicators on pre-earning data
    pre_earning = calculate_indicators_on_data(pre_earning)
    
    # Calculate bar-level range
    pre_earning['bar_range_pct'] = (
        (pre_earning['high'] - pre_earning['low']) / pre_earning['close']
    ) * 100
    
    # Calculate returns (5-min and 30-min)
    pre_earning['ret_5m'] = pre_earning['close'].pct_change(300) * 100
    pre_earning['ret_30m'] = pre_earning['close'].pct_change(1800) * 100
    
    # Baseline statistics
    stats = {
        'ticker': ticker,
        'earnings_events': len(earnings_times),
        'pre_earning_rows': len(pre_earning),
        
        # Volatility (ATR-based)
        'baseline_atr_pct_mean': pre_earning['atr_pct'].mean(),
        'baseline_atr_pct_std': pre_earning['atr_pct'].std(),
        
        # Bar range
        'baseline_range_pct_mean': pre_earning['bar_range_pct'].mean(),
        'baseline_range_pct_std': pre_earning['bar_range_pct'].std(),
        'baseline_range_pct_p95': pre_earning['bar_range_pct'].quantile(0.95),
        
        # Volume
        'baseline_vol_mean': pre_earning['vol'].mean(),
        'baseline_vol_std': pre_earning['vol'].std(),
        'baseline_vol_p25': pre_earning['vol'].quantile(0.25),
        'baseline_vol_p50': pre_earning['vol'].quantile(0.50),
        'baseline_vol_p75': pre_earning['vol'].quantile(0.75),
        
        # RSI
        'baseline_rsi_mean': pre_earning['rsi'].mean(),
        'baseline_rsi_std': pre_earning['rsi'].std(),
        'baseline_rsi_p05': pre_earning['rsi'].quantile(0.05),
        'baseline_rsi_p95': pre_earning['rsi'].quantile(0.95),
        
        # Intraday moves (5-min)
        'baseline_5m_move_pct_mean': pre_earning['ret_5m'].abs().mean(),
        'baseline_5m_move_pct_std': pre_earning['ret_5m'].std(),
        'baseline_5m_move_pct_p95': pre_earning['ret_5m'].abs().quantile(0.95),
        
        # Intraday moves (30-min)
        'baseline_30m_move_pct_mean': pre_earning['ret_30m'].abs().mean(),
        'baseline_30m_move_pct_std': pre_earning['ret_30m'].std(),
        'baseline_30m_move_pct_p95': pre_earning['ret_30m'].abs().quantile(0.95),
        
        # VWAP distance
        'baseline_vwap_dist_pct_mean': pre_earning['vwap_distance_pct'].abs().mean(),
        'baseline_vwap_dist_pct_std': pre_earning['vwap_distance_pct'].std(),
    }
    
    print(f"✓")
    return stats


def main():
    print("=" * 80)
    print("CALCULATING BASELINE STATISTICS FROM PRE-EARNING PERIODS")
    print("=" * 80)
    print(f"Source: Original merged files in {DATA_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)
    print()
    
    results = []
    
    for ticker in QUALITY_TICKERS:
        # Find ticker file
        file_patterns = [
            DATA_DIR / f"{ticker}_second_merged.csv",
            DATA_DIR / f"{ticker}.csv",
            DATA_DIR / f"{ticker}_merged.csv"
        ]
        
        ticker_file = None
        for pattern in file_patterns:
            if pattern.exists():
                ticker_file = pattern
                break
        
        if ticker_file is None:
            print(f"  {ticker}: ❌ File not found")
            continue
        
        # Calculate baseline stats
        stats = calculate_ticker_baseline(ticker_file)
        
        if stats is not None:
            results.append(stats)
    
    # Create DataFrame
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    baseline_df = pd.DataFrame(results)
    
    print(f"Successfully processed: {len(baseline_df)}/{len(QUALITY_TICKERS)} tickers")
    print()
    
    # Save
    baseline_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    
    # Show sample
    print("Sample baseline stats (first 5 tickers):")
    print(baseline_df.head()[['ticker', 'baseline_atr_pct_mean', 'baseline_vol_mean', 
                               'baseline_5m_move_pct_p95', 'baseline_rsi_p95']])
    print()
    
    # Show cross-ticker variation
    print("Cross-ticker variation (proves need for adaptive thresholds):")
    print(f"  ATR% range: {baseline_df['baseline_atr_pct_mean'].min():.4f} - {baseline_df['baseline_atr_pct_mean'].max():.4f}")
    print(f"  5m move p95 range: {baseline_df['baseline_5m_move_pct_p95'].min():.2f}% - {baseline_df['baseline_5m_move_pct_p95'].max():.2f}%")
    print(f"  RSI p95 range: {baseline_df['baseline_rsi_p95'].min():.1f} - {baseline_df['baseline_rsi_p95'].max():.1f}")
    print(f"  Volume mean range: {baseline_df['baseline_vol_mean'].min():,.0f} - {baseline_df['baseline_vol_mean'].max():,.0f}")


if __name__ == '__main__':
    main()
