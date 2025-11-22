#!/usr/bin/env python3
"""
Scan ALL files in processed folder and create consolidated dataset.

Merges useful columns from:
- with_derived (cvd_zscore, delta features)
- with_signals (fixed threshold signals for comparison)
- with_baseline (baseline stats)
- with_adaptive_signals (adaptive signals)
"""

import pandas as pd
from pathlib import Path

# Base directory
PROCESSED_DIR = Path('/Volumes/Extreme SSD/trading_data/stock/data/processed')
OUTPUT_FILE = PROCESSED_DIR / 'earnings_consolidated_final.csv'

# Key for joining
JOIN_KEY = ['ticker', 'timestamp']

def scan_all_files():
    """Scan all CSV files and report columns."""
    print("="*80)
    print("SCANNING ALL FILES IN PROCESSED FOLDER")
    print("="*80)
    
    files = sorted(PROCESSED_DIR.glob('*.csv'))
    
    file_info = {}
    for f in files:
        try:
            df = pd.read_csv(f, nrows=1)
            file_info[f.name] = {
                'path': f,
                'columns': set(df.columns),
                'count': len(df.columns)
            }
            print(f"{f.name:50s} - {len(df.columns):3d} columns")
        except Exception as e:
            print(f"{f.name:50s} - ERROR: {e}")
    
    return file_info


def identify_useful_columns(file_info):
    """Identify columns to merge."""
    print("\n" + "="*80)
    print("IDENTIFYING USEFUL COLUMNS")
    print("="*80)
    
    # Start with current final as base
    base_file = 'earnings_with_adaptive_signals.csv'
    if base_file not in file_info:
        print(f"ERROR: {base_file} not found!")
        return None
    
    base_cols = file_info[base_file]['columns']
    print(f"\nBase: {base_file} ({len(base_cols)} columns)")
    
    # Find additional columns from other files
    additions = {}
    
    for fname, info in file_info.items():
        if fname == base_file:
            continue
        
        new_cols = info['columns'] - base_cols - set(JOIN_KEY)
        if new_cols:
            additions[fname] = new_cols
            print(f"\n{fname}:")
            for col in sorted(new_cols):
                print(f"  + {col}")
    
    return additions


def create_consolidated_dataset(file_info, additions):
    """Create consolidated dataset with all useful columns."""
    print("\n" + "="*80)
    print("CREATING CONSOLIDATED DATASET")
    print("="*80)
    
    # Load base
    base_file = PROCESSED_DIR / 'earnings_with_adaptive_signals.csv'
    print(f"\nLoading base: {base_file.name}...")
    df = pd.read_csv(base_file)
    print(f"  Base: {len(df):,} rows × {len(df.columns)} columns")
    
    # Merge additional columns
    for fname, cols_to_add in additions.items():
        if not cols_to_add:
            continue
        
        source_file = file_info[fname]['path']
        print(f"\nMerging from {fname}...")
        print(f"  Adding {len(cols_to_add)} columns: {', '.join(sorted(list(cols_to_add)[:5]))}...")
        
        # Load only needed columns
        cols_load = JOIN_KEY + list(cols_to_add)
        df_source = pd.read_csv(source_file, usecols=cols_load)
        
        # Merge
        df = df.merge(df_source, on=JOIN_KEY, how='left', suffixes=('', '_dup'))
        
        # Drop any duplicate columns
        dup_cols = [c for c in df.columns if c.endswith('_dup')]
        if dup_cols:
            df = df.drop(columns=dup_cols)
        
        print(f"  After merge: {len(df):,} rows × {len(df.columns)} columns")
    
    return df


def main():
    # Scan all files
    file_info = scan_all_files()
    
    # Identify useful additions
    additions = identify_useful_columns(file_info)
    
    if not additions:
        print("\n✅ No additional columns found - current dataset is complete!")
        return
    
    # Create consolidated dataset
    df = create_consolidated_dataset(file_info, additions)
    
    # Save
    print(f"\n{'-'*80}")
    print(f"Saving consolidated dataset...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"File: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    
    # Show new columns
    base_file = PROCESSED_DIR / 'earnings_with_adaptive_signals.csv'
    df_base = pd.read_csv(base_file, nrows=1)
    new_cols = set(df.columns) - set(df_base.columns)
    
    if new_cols:
        print(f"\n{len(new_cols)} new columns added:")
        for col in sorted(new_cols):
            print(f"  + {col}")


if __name__ == '__main__':
    main()
