#!/usr/bin/env python3
"""
Analyze data quality across all second-bar ticker files.

Usage:
    python analyze_data_quality.py
    python analyze_data_quality.py --limit 5  # Test on first 5 tickers
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

def analyze_ticker_quality(file_path, sample_size=50000):
    """Analyze data quality for a single ticker"""
    
    ticker = Path(file_path).stem.replace('_second_merged', '')
    
    try:
        # Load sample of data
        df = pd.read_csv(file_path, nrows=sample_size)
        
        # Parse timestamp column
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp')
        
        # Calculate gaps
        df['time_gap'] = df['timestamp'].diff().dt.total_seconds()
        
        # Compute metrics
        metrics = {
            'ticker': ticker,
            'sample_bars': len(df),
            'median_gap': df['time_gap'].median(),
            'mean_gap': df['time_gap'].mean(),
            'std_gap': df['time_gap'].std(),
            'min_gap': df['time_gap'].min(),
            'max_gap': df['time_gap'].max(),
            'p25_gap': df['time_gap'].quantile(0.25),
            'p50_gap': df['time_gap'].quantile(0.50),
            'p75_gap': df['time_gap'].quantile(0.75),
            'p90_gap': df['time_gap'].quantile(0.90),
            'p95_gap': df['time_gap'].quantile(0.95),
            'p99_gap': df['time_gap'].quantile(0.99),
            'pct_1s_bars': (df['time_gap'] == 1.0).sum() / len(df) * 100,
            'pct_gaps_over_2s': (df['time_gap'] > 2).sum() / len(df) * 100,
            'pct_gaps_over_5s': (df['time_gap'] > 5).sum() / len(df) * 100,
            'pct_gaps_over_10s': (df['time_gap'] > 10).sum() / len(df) * 100,
            'pct_gaps_over_30s': (df['time_gap'] > 30).sum() / len(df) * 100,
        }
        
        # Calculate bars per second (inverse of median gap)
        metrics['bars_per_second'] = 1.0 / metrics['median_gap'] if metrics['median_gap'] > 0 else 0
        
        # Estimate impact of resampling to 1s
        time_span = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        bars_after_resample = int(time_span)
        metrics['bars_after_1s_resample'] = bars_after_resample
        metrics['resample_inflation_pct'] = (bars_after_resample / len(df) - 1) * 100
        metrics['artificial_data_pct'] = ((bars_after_resample - len(df)) / bars_after_resample) * 100
        
        # Classify liquidity
        if metrics['bars_per_second'] >= 0.7:
            metrics['liquidity'] = 'HIGH'
            metrics['recommended_rsi_period'] = 14
            metrics['quality_score'] = 9  # 0-10 scale
        elif metrics['bars_per_second'] >= 0.3:
            metrics['liquidity'] = 'MEDIUM'
            metrics['recommended_rsi_period'] = 10
            metrics['quality_score'] = 6
        else:
            metrics['liquidity'] = 'LOW'
            metrics['recommended_rsi_period'] = 7
            metrics['quality_score'] = 3
        
        # Recommendation for inclusion
        if metrics['pct_1s_bars'] >= 60 and metrics['pct_gaps_over_30s'] < 5:
            metrics['recommendation'] = 'INCLUDE'
        elif metrics['pct_1s_bars'] >= 30 and metrics['pct_gaps_over_30s'] < 15:
            metrics['recommendation'] = 'REVIEW'
        else:
            metrics['recommendation'] = 'EXCLUDE'
        
        return metrics, None
        
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Analyze data quality for all tickers")
    parser.add_argument(
        '--input-dir',
        default='/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv',
        help='Directory containing *_second_merged.csv files'
    )
    parser.add_argument(
        '--output',
        default='data_quality_report.csv',
        help='Output CSV file for quality metrics'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit analysis to first N tickers (for testing)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50000,
        help='Number of rows to sample per ticker (default: 50000)'
    )
    
    args = parser.parse_args()
    
    # Find all ticker files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    ticker_files = sorted(input_path.glob('*_second_merged.csv'))
    
    # Filter out macOS system files
    ticker_files = [f for f in ticker_files if not f.name.startswith('._')]
    
    if not ticker_files:
        print(f"No *_second_merged.csv files found in {args.input_dir}")
        return 1
    
    if args.limit:
        ticker_files = ticker_files[:args.limit]
    
    print(f"Analyzing {len(ticker_files)} tickers...")
    print(f"Sample size: {args.sample_size:,} rows per ticker")
    print("=" * 80)
    
    results = []
    errors = []
    
    for i, file_path in enumerate(ticker_files, 1):
        ticker = file_path.stem.replace('_second_merged', '')
        print(f"[{i}/{len(ticker_files)}] {ticker}...", end=' ')
        
        metrics, error = analyze_ticker_quality(str(file_path), args.sample_size)
        
        if metrics:
            results.append(metrics)
            print(f"{metrics['liquidity']} (gap: {metrics['median_gap']:.2f}s)")
        else:
            errors.append({'ticker': ticker, 'error': error})
            print(f"ERROR: {error}")
    
    # Create summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Save to CSV
        df_results.to_csv(args.output, index=False)
        print(f"\nFull report saved to: {args.output}")
        
        # Print summary stats
        print(f"\nLiquidity Distribution:")
        print(df_results['liquidity'].value_counts().to_string())
        
        print(f"\nGap Statistics by Liquidity:")
        gap_stats = df_results.groupby('liquidity')[['median_gap', 'p95_gap', 'pct_1s_bars', 'pct_gaps_over_30s']].agg(['mean', 'min', 'max'])
        print(gap_stats.to_string())
        
        print(f"\nResampling Impact:")
        resample_stats = df_results.groupby('liquidity')[['resample_inflation_pct', 'artificial_data_pct']].mean()
        print(resample_stats.to_string())
        
        # Recommendations
        print(f"\n{'=' * 80}")
        print("TICKER RECOMMENDATIONS")
        print("=" * 80)
        
        include = df_results[df_results['recommendation'] == 'INCLUDE']
        review = df_results[df_results['recommendation'] == 'REVIEW']
        exclude = df_results[df_results['recommendation'] == 'EXCLUDE']
        
        print(f"\n✅ INCLUDE ({len(include)} tickers): High quality, suitable for resampling")
        if len(include) > 0:
            print(f"   Tickers: {', '.join(include['ticker'].tolist())}")
            print(f"   Avg 1s bars: {include['pct_1s_bars'].mean():.1f}%")
            print(f"   Avg resample inflation: {include['resample_inflation_pct'].mean():.1f}%")
        
        print(f"\n⚠️  REVIEW ({len(review)} tickers): Medium quality, acceptable with caveats")
        if len(review) > 0:
            print(f"   Tickers: {', '.join(review['ticker'].tolist())}")
            print(f"   Avg 1s bars: {review['pct_1s_bars'].mean():.1f}%")
            print(f"   Avg resample inflation: {review['resample_inflation_pct'].mean():.1f}%")
        
        print(f"\n❌ EXCLUDE ({len(exclude)} tickers): Low quality, not suitable")
        if len(exclude) > 0:
            print(f"   Tickers: {', '.join(exclude['ticker'].tolist())}")
            print(f"   Avg 1s bars: {exclude['pct_1s_bars'].mean():.1f}%")
            print(f"   Avg resample inflation: {exclude['resample_inflation_pct'].mean():.1f}%")
            print(f"   Reason: High artificial data % after resampling")
        
        # Specific problem tickers
        print(f"\n{'=' * 80}")
        print("DETAILED ISSUES")
        print("=" * 80)
        
        # Tickers with excessive gaps
        excessive_gaps = df_results[df_results['pct_gaps_over_30s'] > 10]
        if len(excessive_gaps) > 0:
            print(f"\n⚠️  Tickers with >10% gaps over 30 seconds:")
            for _, row in excessive_gaps.iterrows():
                print(f"   {row['ticker']}: {row['pct_gaps_over_30s']:.1f}% gaps >30s, max gap {row['max_gap']:.0f}s")
        
        # Tickers with massive resample inflation
        high_inflation = df_results[df_results['resample_inflation_pct'] > 200]
        if len(high_inflation) > 0:
            print(f"\n⚠️  Tickers with >200% resample inflation:")
            for _, row in high_inflation.iterrows():
                print(f"   {row['ticker']}: {row['resample_inflation_pct']:.1f}% inflation, {row['artificial_data_pct']:.1f}% artificial data")
        
        print(f"\n{'=' * 80}")
        print(f"Final recommendation: Use {len(include)} INCLUDE tickers for analysis")
        print(f"                      Review {len(review)} tickers case-by-case")
        print(f"                      Exclude {len(exclude)} tickers due to poor data quality")
        print("=" * 80)
    
    if errors:
        print(f"\n\nERRORS ({len(errors)}):")
        for err in errors:
            print(f"  {err['ticker']}: {err['error']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
