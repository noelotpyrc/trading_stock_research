#!/usr/bin/env python3
"""
Analyze data quality specifically during earnings events (0-30 mins post-earnings).

This script filters to earnings event windows before analyzing gap distributions,
since liquidity characteristics may differ significantly during earnings vs regular trading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os

def load_earnings_from_csv(ticker_file):
    """Extract unique earnings datetimes from a ticker's second-bar CSV"""
    df = pd.read_csv(ticker_file, usecols=['acceptance_datetime_utc'])
    earnings_times = pd.to_datetime(df['acceptance_datetime_utc'], utc=True).unique()
    return list(earnings_times)

def analyze_ticker_gaps_at_earnings(ticker_file):
    """
    Analyze gap distribution during 0-30 min post-earnings windows.
    
    Args:
        ticker_file: Path to ticker's second-bar data (with acceptance_datetime_utc column)
    
    Returns:
        metrics dict, error string (if any)
    """
    ticker = Path(ticker_file).stem.replace('_second_merged', '')
    
    try:
        # Load data
        df = pd.read_csv(ticker_file, nrows=None)  # Load all data
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract unique earnings times
        earnings_times = df['acceptance_datetime_utc'].unique()
        
        # Filter to earnings windows (0-30 min post-event)
        earnings_mask = pd.Series(False, index=df.index)
        
        for event_time in earnings_times:
            window_start = event_time
            window_end = event_time + pd.Timedelta(minutes=30)
            
            mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
            earnings_mask |= mask
        
        df_earnings = df[earnings_mask].copy()
        
        if len(df_earnings) < 100:
            return None, f"Insufficient data during earnings windows ({len(df_earnings)} bars)"
        
        # Calculate time gaps
        df_earnings['time_gap'] = df_earnings['timestamp'].diff().dt.total_seconds()
        df_earnings = df_earnings[df_earnings['time_gap'].notna()]
        
        if len(df_earnings) == 0:
            return None, "No valid gaps found"
        
        # Compute metrics
        metrics = {
            'ticker': ticker,
            'num_earnings_events': len(earnings_times),
            'total_bars_in_windows': len(df_earnings),
            'median_gap': df_earnings['time_gap'].median(),
            'mean_gap': df_earnings['time_gap'].mean(),
            'std_gap': df_earnings['time_gap'].std(),
            'min_gap': df_earnings['time_gap'].min(),
            'max_gap': df_earnings['time_gap'].max(),
            'p25_gap': df_earnings['time_gap'].quantile(0.25),
            'p50_gap': df_earnings['time_gap'].quantile(0.50),
            'p75_gap': df_earnings['time_gap'].quantile(0.75),
            'p90_gap': df_earnings['time_gap'].quantile(0.90),
            'p95_gap': df_earnings['time_gap'].quantile(0.95),
            'p99_gap': df_earnings['time_gap'].quantile(0.99),
            'pct_1s_bars': (df_earnings['time_gap'] == 1.0).sum() / len(df_earnings) * 100,
            'pct_gaps_over_2s': (df_earnings['time_gap'] > 2).sum() / len(df_earnings) * 100,
            'pct_gaps_over_5s': (df_earnings['time_gap'] > 5).sum() / len(df_earnings) * 100,
            'pct_gaps_over_10s': (df_earnings['time_gap'] > 10).sum() / len(df_earnings) * 100,
            'pct_gaps_over_30s': (df_earnings['time_gap'] > 30).sum() / len(df_earnings) * 100,
        }
        
        # Calculate bars per second
        metrics['bars_per_second'] = 1.0 / metrics['median_gap'] if metrics['median_gap'] > 0 else 0
        
        # Estimate impact of resampling to 1s (for 30-min window = 1800 seconds)
        window_duration_sec = 30 * 60  # 30 minutes
        expected_bars_per_event = window_duration_sec
        total_bars_after_resample = expected_bars_per_event * metrics['num_earnings_events']
        
        metrics['bars_after_1s_resample'] = total_bars_after_resample
        metrics['resample_inflation_pct'] = (total_bars_after_resample / metrics['total_bars_in_windows'] - 1) * 100
        metrics['artificial_data_pct'] = ((total_bars_after_resample - metrics['total_bars_in_windows']) / total_bars_after_resample) * 100
        
        # Classify liquidity based on earnings-period data
        if metrics['bars_per_second'] >= 0.7:
            metrics['liquidity'] = 'HIGH'
            metrics['recommended_rsi_period'] = 14
            metrics['quality_score'] = 9
        elif metrics['bars_per_second'] >= 0.3:
            metrics['liquidity'] = 'MEDIUM'
            metrics['recommended_rsi_period'] = 10
            metrics['quality_score'] = 6
        else:
            metrics['liquidity'] = 'LOW'
            metrics['recommended_rsi_period'] = 7
            metrics['quality_score'] = 3
        
        # Recommendation for inclusion (stricter for earnings data)
        if metrics['pct_1s_bars'] >= 50 and metrics['pct_gaps_over_30s'] < 10:
            metrics['recommendation'] = 'INCLUDE'
        elif metrics['pct_1s_bars'] >= 20 and metrics['pct_gaps_over_30s'] < 20:
            metrics['recommendation'] = 'REVIEW'
        else:
            metrics['recommendation'] = 'EXCLUDE'
        
        return metrics, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Analyze data quality during earnings events (0-30 min post-event)')
    parser.add_argument('--data-dir', default='/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv',
                        help='Directory containing second-bar CSV files with earnings metadata')
    parser.add_argument('--output', default='earnings_data_quality_report.csv',
                        help='Output CSV file for results')
    
    args = parser.parse_args()
    
    # Find all ticker files
    data_files = sorted(Path(args.data_dir).glob('*_second_merged.csv'))
    
    # Filter out macOS system files
    data_files = [f for f in data_files if not f.name.startswith('._')]
    
    print(f"\nAnalyzing {len(data_files)} tickers during earnings events...")
    print("Time window: 0-30 minutes post-earnings")
    print("=" * 80)
    
    results = []
    errors = []
    
    for i, file_path in enumerate(data_files, 1):
        ticker = file_path.stem.replace('_second_merged', '')
        
        metrics, error = analyze_ticker_gaps_at_earnings(str(file_path))
        
        if error:
            errors.append((ticker, error))
            print(f"[{i}/{len(data_files)}] {ticker}... ERROR: {error}")
        else:
            results.append(metrics)
            print(f"[{i}/{len(data_files)}] {ticker}... {metrics['liquidity']} "
                  f"(gap: {metrics['median_gap']:.2f}s, {metrics['total_bars_in_windows']} bars, "
                  f"{metrics['num_earnings_events']} events)")
    
    # Create summary
    print("\n" + "=" * 80)
    print("SUMMARY - EARNINGS EVENT DATA QUALITY (0-30 min post-earnings)")
    print("=" * 80)
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Save to CSV
        df_results.to_csv(args.output, index=False)
        print(f"\nFull report saved to: {args.output}")
        
        # Print summary stats
        print(f"\nLiquidity Distribution (During Earnings):")
        print(df_results['liquidity'].value_counts().to_string())
        
        print(f"\nGap Statistics by Liquidity (Earnings Windows):")
        gap_stats = df_results.groupby('liquidity')[['median_gap', 'p95_gap', 'pct_1s_bars', 'pct_gaps_over_30s']].agg(['mean', 'min', 'max'])
        print(gap_stats.to_string())
        
        print(f"\nResampling Impact (30-min windows per event):")
        resample_stats = df_results.groupby('liquidity')[['resample_inflation_pct', 'artificial_data_pct']].mean()
        print(resample_stats.to_string())
        
        # Recommendations
        print(f"\n{'=' * 80}")
        print("TICKER RECOMMENDATIONS (EARNINGS-SPECIFIC)")
        print("=" * 80)
        
        include = df_results[df_results['recommendation'] == 'INCLUDE']
        review = df_results[df_results['recommendation'] == 'REVIEW']
        exclude = df_results[df_results['recommendation'] == 'EXCLUDE']
        
        print(f"\n✅ INCLUDE ({len(include)} tickers): High quality during earnings")
        if len(include) > 0:
            print(f"   Tickers: {', '.join(include['ticker'].tolist())}")
            print(f"   Avg 1s bars: {include['pct_1s_bars'].mean():.1f}%")
            print(f"   Avg bars per event: {(include['total_bars_in_windows'] / include['num_earnings_events']).mean():.0f}")
        
        print(f"\n⚠️  REVIEW ({len(review)} tickers): Medium quality during earnings")
        if len(review) > 0:
            print(f"   Tickers: {', '.join(review['ticker'].tolist())}")
            print(f"   Avg 1s bars: {review['pct_1s_bars'].mean():.1f}%")
            print(f"   Avg bars per event: {(review['total_bars_in_windows'] / review['num_earnings_events']).mean():.0f}")
        
        print(f"\n❌ EXCLUDE ({len(exclude)} tickers): Poor quality during earnings windows")
        if len(exclude) > 0:
            print(f"   Tickers: {', '.join(exclude['ticker'].tolist())}")
            print(f"   Avg 1s bars: {exclude['pct_1s_bars'].mean():.1f}%")
            print(f"   Reason: Sparse data during critical 30-min post-earnings window")
        
        # Comparison with overall data quality
        print(f"\n{'=' * 80}")
        print("NOTE: This analysis focuses ONLY on 0-30 min post-earnings windows.")
        print("Data quality during earnings may differ from regular trading hours.")
        print("Pre/post-market earnings will show lower liquidity than market-hour earnings.")
        print("=" * 80)
    
    if errors:
        print(f"\n\nERRORS ({len(errors)}):")
        for ticker, error in errors:
            print(f"  {ticker}: {error}")

if __name__ == '__main__':
    main()
