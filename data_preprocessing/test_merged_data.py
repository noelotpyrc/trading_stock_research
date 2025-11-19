import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Spot check merged OHLCV data.")
    parser.add_argument("ticker", type=str, help="Stock ticker (e.g., AAPL)")
    parser.add_argument("type", type=str, choices=["minute", "second"], help="Data type (minute or second)")
    parser.add_argument("timestamp", type=str, help="Timestamp in UTC (YYYY-MM-DD HH:MM:SS)")
    
    args = parser.parse_args()
    
    # Construct file path
    filename = f"{args.ticker}_{args.type}_merged.csv"
    filepath = os.path.join("/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv", filename)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    print(f"Loading {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Convert input timestamp to nanoseconds
    try:
        # Assume input is UTC
        ts = pd.Timestamp(args.timestamp, tz='UTC')
        target_ns = ts.value
        print(f"Searching for timestamp: {args.timestamp} (UTC) -> {target_ns} (ns)")
    except Exception as e:
        print(f"Error parsing timestamp: {e}")
        return

    # Find the row
    match = df[df['timestamp'] == target_ns]
    
    if match.empty:
        print("No data found for this timestamp.")
    else:
        print("\nMatch found:")
        row = match.iloc[0]
        print(f"  Acceptance Datetime (UTC): {row.get('acceptance_datetime_utc', 'N/A')}")
        print(f"  EPS Estimate:              {row.get('EPS Estimate', 'N/A')}")
        print(f"  Reported EPS:              {row.get('Reported EPS', 'N/A')}")
        print(f"  Surprise(%):               {row.get('Surprise(%)', 'N/A')}")
        print("-" * 30)
        print(f"  Full Row Data:\n{row}")

if __name__ == "__main__":
    main()
