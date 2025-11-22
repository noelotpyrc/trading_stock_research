#!/usr/bin/env python3
"""
Batch process all second-bar merged files to add earnings signal metadata.

This script:
1. Finds all *_second_merged.csv files in the input directory
2. Labels event windows for each file
3. Calculates earnings signals and targets
4. Saves processed files to output directory

Usage:
    python batch_process_earnings.py
    python batch_process_earnings.py --input-dir /path/to/input --output-dir /path/to/output
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering.label_earning_event_windows import add_features as label_windows
from feature_engineering.add_earnings_signals_meta import add_earnings_signals


def process_file(input_file, output_dir, temp_dir):
    """
    Process a single file through the pipeline.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory for final output
        temp_dir: Directory for intermediate files
        
    Returns:
        tuple: (success: bool, message: str)
    """
    ticker = Path(input_file).stem.replace('_second_merged', '')
    
    try:
        # Step 1: Label windows
        print(f"  [1/2] Labeling event windows...")
        temp_labeled = os.path.join(temp_dir, f"{ticker}_labeled.csv")
        label_windows(input_file, temp_labeled)
        
        # Step 2: Calculate signals
        print(f"  [2/2] Calculating earnings signals...")
        output_file = os.path.join(output_dir, f"{ticker}_earnings_signals.csv")
        add_earnings_signals(temp_labeled, output_file, pre_labeled=True)
        
        # Cleanup temp file
        if os.path.exists(temp_labeled):
            os.remove(temp_labeled)
            
        return True, f"Successfully processed and saved to {output_file}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Batch process earnings data for all tickers")
    parser.add_argument(
        '--input-dir',
        default='/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv',
        help='Directory containing *_second_merged.csv files'
    )
    parser.add_argument(
        '--output-dir',
        default='/Volumes/Extreme SSD/trading_data/stock/data/earnings_signals',
        help='Directory for processed output files'
    )
    parser.add_argument(
        '--temp-dir',
        default='./temp',
        help='Directory for temporary intermediate files'
    )
    parser.add_argument(
        '--pattern',
        default='*_second_merged.csv',
        help='File pattern to match (default: *_second_merged.csv)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that have already been processed'
    )
    
    args = parser.parse_args()
    
    # Create output and temp directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Find all input files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
        
    input_files = sorted(input_path.glob(args.pattern))
    
    if not input_files:
        print(f"No files found matching pattern '{args.pattern}' in {args.input_dir}")
        return 1
    
    # Apply limit if specified
    if args.limit:
        input_files = input_files[:args.limit]
    
    print(f"Found {len(input_files)} files to process")
    print(f"Output directory: {args.output_dir}")
    print(f"=" * 80)
    
    # Process each file
    results = []
    start_time = time.time()
    
    for i, input_file in enumerate(input_files, 1):
        ticker = input_file.stem.replace('_second_merged', '')
        output_file = os.path.join(args.output_dir, f"{ticker}_earnings_signals.csv")
        
        # Skip if already exists
        if args.skip_existing and os.path.exists(output_file):
            print(f"[{i}/{len(input_files)}] {ticker}: SKIPPED (already exists)")
            results.append((ticker, 'skipped', 'Already processed'))
            continue
        
        print(f"\n[{i}/{len(input_files)}] Processing {ticker}...")
        file_start = time.time()
        
        success, message = process_file(str(input_file), args.output_dir, args.temp_dir)
        
        file_duration = time.time() - file_start
        status = 'SUCCESS' if success else 'FAILED'
        
        print(f"  Status: {status} ({file_duration:.1f}s)")
        if not success:
            print(f"  {message}")
        
        results.append((ticker, status, message))
    
    # Summary
    total_duration = time.time() - start_time
    success_count = sum(1 for _, status, _ in results if status == 'SUCCESS')
    failed_count = sum(1 for _, status, _ in results if status == 'FAILED')
    skipped_count = sum(1 for _, status, _ in results if status == 'skipped')
    
    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total files: {len(input_files)}")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"Duration: {total_duration:.1f}s")
    
    # Show failures
    if failed_count > 0:
        print(f"\nFailed files:")
        for ticker, status, message in results:
            if status == 'FAILED':
                print(f"  {ticker}: {message}")
    
    # Save log
    log_file = os.path.join(args.output_dir, f"batch_process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, 'w') as f:
        f.write(f"Batch Processing Log - {datetime.now()}\n")
        f.write(f"Input: {args.input_dir}\n")
        f.write(f"Output: {args.output_dir}\n")
        f.write(f"Total: {len(input_files)}, Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}\n\n")
        for ticker, status, message in results:
            f.write(f"{ticker}: {status} - {message}\n")
    
    print(f"\nLog saved to: {log_file}")
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
