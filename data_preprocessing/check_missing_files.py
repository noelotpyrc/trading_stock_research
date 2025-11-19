import pandas as pd
import os

# Configuration
DATA_ROOT = "/Volumes/Extreme SSD/trading_data/stock/data/ltf_ohlcv"
METADATA_PATH = os.path.join(DATA_ROOT, "metadata", "metadata.csv")

def main():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    print(f"Reading metadata from {METADATA_PATH}...")
    try:
        metadata = pd.read_csv(METADATA_PATH)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    missing_files = []
    total_files_checked = 0

    print("Checking for missing files...")

    for index, row in metadata.iterrows():
        # Check minute file
        minute_file = row.get('minute_file')
        if pd.notna(minute_file):
            total_files_checked += 1
            minute_path = os.path.join(DATA_ROOT, minute_file)
            if not os.path.exists(minute_path):
                missing_files.append(f"Row {index} ({row.get('ticker')}): Minute file missing - {minute_file}")

        # Check second files
        for i in range(1, 4):
            second_file = row.get(f'second_file_{i}')
            if pd.notna(second_file):
                total_files_checked += 1
                second_path = os.path.join(DATA_ROOT, second_file)
                if not os.path.exists(second_path):
                    missing_files.append(f"Row {index} ({row.get('ticker')}): Second file {i} missing - {second_file}")

    print(f"\nCheck complete.")
    print(f"Total files checked: {total_files_checked}")
    print(f"Missing files found: {len(missing_files)}")

    if missing_files:
        print("\nMissing Files List:")
        for msg in missing_files:
            print(msg)
    else:
        print("\nAll files found!")

if __name__ == "__main__":
    main()
