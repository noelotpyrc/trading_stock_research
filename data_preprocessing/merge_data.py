import pandas as pd
import os
import glob

# Configuration
DATA_ROOT = "/Volumes/Extreme SSD/trading_data/stock/data/ltf_ohlcv"
METADATA_PATH = os.path.join(DATA_ROOT, "metadata", "metadata.csv")
OUTPUT_DIR = "/Volumes/Extreme SSD/trading_data/stock/data/merged_ohlcv"

def load_metadata():
    """Loads metadata from the CSV file."""
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    return pd.read_csv(METADATA_PATH)

def process_ticker(ticker, group, output_dir):
    """Processes a single ticker's data."""
    minute_dfs = []
    second_dfs = []

    print(f"Processing {ticker}...")

    for _, row in group.iterrows():
        # Extract metadata columns to add
        meta_cols = {
            'ticker': ticker,
            'acceptance_datetime_utc': row.get('acceptance_datetime_utc'),
            'EPS Estimate': row.get('EPS Estimate'),
            'Reported EPS': row.get('Reported EPS'),
            'Surprise(%)': row.get('Surprise(%)')
        }

        # --- Process Minute Data ---
        minute_file = row.get('minute_file')
        if pd.notna(minute_file):
            minute_path = os.path.join(DATA_ROOT, minute_file)
            if os.path.exists(minute_path):
                try:
                    df = pd.read_csv(minute_path)
                    for col, val in meta_cols.items():
                        df[col] = val
                    minute_dfs.append(df)
                except Exception as e:
                    print(f"  Error reading minute file {minute_file}: {e}")
            else:
                print(f"  Warning: Minute file not found: {minute_path}")

        # --- Process Second Data ---
        for i in range(1, 4):
            second_file = row.get(f'second_file_{i}')
            if pd.notna(second_file):
                second_path = os.path.join(DATA_ROOT, second_file)
                if os.path.exists(second_path):
                    try:
                        df = pd.read_csv(second_path)
                        for col, val in meta_cols.items():
                            df[col] = val
                        second_dfs.append(df)
                    except Exception as e:
                        print(f"  Error reading second file {second_file}: {e}")
                else:
                    print(f"  Warning: Second file not found: {second_path}")

    # --- Save Merged Data ---
    os.makedirs(output_dir, exist_ok=True)

    if minute_dfs:
        merged_minute = pd.concat(minute_dfs, ignore_index=True)
        output_path = os.path.join(output_dir, f"{ticker}_minute_merged.csv")
        merged_minute.to_csv(output_path, index=False)
        print(f"  Saved minute data to {output_path}")
    else:
        print(f"  No minute data found for {ticker}")

    if second_dfs:
        merged_second = pd.concat(second_dfs, ignore_index=True)
        output_path = os.path.join(output_dir, f"{ticker}_second_merged.csv")
        merged_second.to_csv(output_path, index=False)
        print(f"  Saved second data to {output_path}")
    else:
        print(f"  No second data found for {ticker}")

def main():
    try:
        metadata = load_metadata()
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        return

    # Group by ticker
    grouped = metadata.groupby('ticker')

    for ticker, group in grouped:
        process_ticker(ticker, group, OUTPUT_DIR)

if __name__ == "__main__":
    main()
