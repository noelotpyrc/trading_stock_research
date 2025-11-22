#!/usr/bin/env python3
"""
Comprehensive data audit before analysis.

Checks:
1. Data completeness (tickers, time range, events)
2. Feature inventory and NaN rates
3. Intermediate files for missing useful columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# File paths
FINAL_FILE = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_with_adaptive_signals.csv')
INTERMEDIATE_FILES = {
    'original': Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_final.csv'),
    'with_baseline': Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_baseline.csv'),
    'with_derived': Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_derived.csv'),
    'with_signals': Path('/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_quality_tickers_with_signals.csv'),
}

OUTPUT_FILE = Path('data/audit/data_audit_report.md')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def audit_completeness(df):
    """Audit data completeness."""
    print("="*80)
    print("1. DATA COMPLETENESS")
    print("="*80)
    
    report = []
    report.append("# Data Completeness\n")
    
    # Tickers
    tickers = df['ticker'].unique()
    report.append(f"## Tickers: {len(tickers)}\n")
    report.append(f"```\n{', '.join(sorted(tickers))}\n```\n")
    
    # Time range
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'])
    
    time_start = df['timestamp'].min()
    time_end = df['timestamp'].max()
    report.append(f"## Time Range\n")
    report.append(f"- Start: {time_start}\n")
    report.append(f"- End: {time_end}\n")
    report.append(f"- Duration: {(time_end - time_start).days} days\n\n")
    
    # Earnings events
    events = df['acceptance_datetime_utc'].nunique()
    report.append(f"## Earnings Events: {events}\n")
    
    # Events per ticker
    events_per_ticker = df.groupby('ticker')['acceptance_datetime_utc'].nunique()
    report.append(f"- Mean events/ticker: {events_per_ticker.mean():.1f}\n")
    report.append(f"- Min events/ticker: {events_per_ticker.min()}\n")
    report.append(f"- Max events/ticker: {events_per_ticker.max()}\n\n")
    
    # Rows per ticker
    rows_per_ticker = df.groupby('ticker').size()
    report.append(f"## Rows per Ticker\n")
    report.append(f"- Total rows: {len(df):,}\n")
    report.append(f"- Mean rows/ticker: {rows_per_ticker.mean():,.0f}\n")
    report.append(f"- Min rows/ticker: {rows_per_ticker.min():,}\n")
    report.append(f"- Max rows/ticker: {rows_per_ticker.max():,}\n\n")
    
    # Distribution table
    report.append("### Per-Ticker Summary\n\n")
    report.append("| Ticker | Events | Rows | Avg Rows/Event |\n")
    report.append("|--------|--------|------|----------------|\n")
    
    for ticker in sorted(tickers):
        ticker_df = df[df['ticker'] == ticker]
        n_events = ticker_df['acceptance_datetime_utc'].nunique()
        n_rows = len(ticker_df)
        avg_rows = n_rows / n_events if n_events > 0 else 0
        report.append(f"| {ticker} | {n_events} | {n_rows:,} | {avg_rows:.0f} |\n")
    
    report.append("\n")
    
    print(f"  Tickers: {len(tickers)}")
    print(f"  Events: {events}")
    print(f"  Rows: {len(df):,}")
    print(f"  Time range: {time_start.date()} to {time_end.date()}")
    
    return ''.join(report)


def audit_features(df):
    """Audit feature inventory."""
    print("\n" + "="*80)
    print("2. FEATURE INVENTORY")
    print("="*80)
    
    report = []
    report.append("# Feature Inventory\n")
    report.append(f"**Total columns**: {len(df.columns)}\n\n")
    
    # Categorize columns
    categories = {
        'Price OHLCV': ['open', 'high', 'low', 'close', 'vw', 'vol', 'num'],
        'Time': ['timestamp', 'acceptance_datetime_utc', 'seconds_since_event'],
        'Metadata': ['ticker', 'EPS Estimate', 'Reported EPS', 'Surprise(%)'],
        'Event Windows': [c for c in df.columns if c.endswith(('_before', '_after', '_plus_minus_30m'))],
        'Event Signals': ['event_price', 'or_high', 'or_low', 'or_width', 'cvd_since_event', 
                         'vwap_since_event', 'impulse_high', 'impulse_low', 'impulse_range_pct',
                         'first_5m_high', 'first_5m_low'],
        'Normalized Features': [c for c in df.columns if any(x in c for x in ['_pct', 'distance', 'above_ema'])],
        'Technical Indicators': ['rsi', 'roc', 'atr', 'atr_pct', 'vol_ratio'] + 
                               [c for c in df.columns if 'ema' in c.lower()],
        'Baseline Stats': [c for c in df.columns if c.startswith('baseline_')],
        'Targets': [c for c in df.columns if c.startswith('target_')],
        'Fixed Signals': [c for c in df.columns if c.startswith('signal_') and 'adaptive' not in c],
        'Adaptive Signals': [c for c in df.columns if c.startswith('signal_') and 'adaptive' in c],
        'Derived Features': ['cvd_zscore', 'delta', 'delta_20s', 'delta_10s'],
    }
    
    # Create feature summary
    for category, patterns in categories.items():
        cols = [c for c in df.columns if c in patterns or any(p in c for p in patterns if isinstance(p, str) and '*' not in p)]
        cols = list(set(cols))  # Remove duplicates
        
        if cols:
            report.append(f"## {category} ({len(cols)} columns)\n\n")
            
            # NaN statistics
            for col in sorted(cols):
                nan_count = df[col].isna().sum()
                nan_pct = (nan_count / len(df)) * 100
                dtype = df[col].dtype
                report.append(f"- `{col}` ({dtype}) - {nan_pct:.1f}% NaN\n")
            
            report.append("\n")
    
    # Find uncategorized columns
    all_categorized = set()
    for patterns in categories.values():
        for c in df.columns:
            if c in patterns or any(p in c for p in patterns if isinstance(p, str) and '*' not in p):
                all_categorized.add(c)
    
    uncategorized = set(df.columns) - all_categorized
    if uncategorized:
        report.append(f"## Uncategorized ({len(uncategorized)} columns)\n\n")
        for col in sorted(uncategorized):
            nan_count = df[col].isna().sum()
            nan_pct = (nan_count / len(df)) * 100
            report.append(f"- `{col}` - {nan_pct:.1f}% NaN\n")
        report.append("\n")
    
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Baseline stats: {len([c for c in df.columns if c.startswith('baseline_')])}")
    print(f"  Signals: {len([c for c in df.columns if c.startswith('signal_')])}")
    print(f"  Targets: {len([c for c in df.columns if c.startswith('target_')])}")
    
    return ''.join(report)


def audit_intermediate_files():
    """Compare intermediate files for missing columns."""
    print("\n" + "="*80)
    print("3. INTERMEDIATE FILE SCAN")
    print("="*80)
    
    report = []
    report.append("# Intermediate File Comparison\n\n")
    
    # Load column names from each file
    file_columns = {}
    for name, path in INTERMEDIATE_FILES.items():
        if path.exists():
            # Just read first row to get columns
            df_sample = pd.read_csv(path, nrows=1)
            file_columns[name] = set(df_sample.columns)
            print(f"  {name}: {len(df_sample.columns)} columns")
        else:
            print(f"  {name}: FILE NOT FOUND")
            file_columns[name] = set()
    
    # Load final file columns
    df_final = pd.read_csv(FINAL_FILE, nrows=1)
    final_cols = set(df_final.columns)
    print(f"  final: {len(final_cols)} columns")
    
    # Find columns in intermediates but not in final
    report.append("## Columns in Intermediate Files Not in Final\n\n")
    
    found_missing = False
    for name, cols in file_columns.items():
        missing = cols - final_cols
        if missing:
            found_missing = True
            report.append(f"### From `{name}` ({len(missing)} columns)\n\n")
            for col in sorted(missing):
                report.append(f"- `{col}`\n")
            report.append("\n")
    
    if not found_missing:
        report.append("✅ No columns missing from final dataset\n\n")
    
    # Column evolution
    report.append("## Column Count Evolution\n\n")
    report.append("| File | Columns |\n")
    report.append("|------|--------|\n")
    for name in ['original', 'with_baseline', 'with_derived', 'with_signals', 'final']:
        if name == 'final':
            count = len(final_cols)
        else:
            count = len(file_columns.get(name, []))
        report.append(f"| {name} | {count} |\n")
    
    return ''.join(report)


def main():
    print("DATA AUDIT AND SANITY CHECK")
    print("="*80)
    
    # Load final dataset
    print(f"Loading {FINAL_FILE.name}...")
    df = pd.read_csv(FINAL_FILE)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns\n")
    
    # Run audits
    report_parts = []
    report_parts.append(f"# Data Audit Report\n")
    report_parts.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    report_parts.append(f"**Dataset**: `{FINAL_FILE.name}`\n\n")
    report_parts.append("---\n\n")
    
    report_parts.append(audit_completeness(df))
    report_parts.append(audit_features(df))
    report_parts.append(audit_intermediate_files())
    
    # Save report
    report = ''.join(report_parts)
    OUTPUT_FILE.write_text(report)
    
    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)
    print(f"Report saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
