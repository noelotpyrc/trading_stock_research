#!/usr/bin/env python3
"""
Utility script to append short-horizon forward returns (10s, 20s)
to the consolidated earnings dataset without re-running the full pipeline.

Usage:
    python feature_engineering/add_short_horizon_targets.py \
        --input "/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_final.csv" \
        --output "/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_final.csv"

By default the script overwrites the input file; supply a different --output
path if you prefer to create a patched copy first.
"""

import argparse
from pathlib import Path

import pandas as pd

from feature_engineering.targets import get_forward_returns

# Default locations mirror the rest of the pipeline.
DEFAULT_DATASET = Path("/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_consolidated_final.csv")
SHORT_HORIZONS = [10, 20]


def compute_short_targets(group_df: pd.DataFrame, tolerance_seconds: int) -> pd.DataFrame:
    """Return short-horizon targets for a single ticker group."""
    sorted_df = group_df.sort_values("timestamp").copy()
    returns = get_forward_returns(
        sorted_df[["timestamp", "close"]].copy(),
        horizons_seconds=SHORT_HORIZONS,
        tolerance_seconds=tolerance_seconds,
    )
    # Align with the original indices so concatenation preserves row order.
    returns.index = sorted_df.index
    return returns


def patch_dataset(input_path: Path, output_path: Path, tolerance_seconds: int):
    """Load the dataset, compute new targets ticker-by-ticker, and save."""
    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    required = {"timestamp", "close", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {', '.join(sorted(missing))}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print("Computing short-horizon targets (per ticker)...")
    target_frames = []
    for ticker, ticker_df in df.groupby("ticker", group_keys=False):
        if ticker_df.empty:
            continue
        targets = compute_short_targets(ticker_df, tolerance_seconds)
        target_frames.append(targets)

    if not target_frames:
        raise RuntimeError("No ticker groups were processed; aborting.")

    new_targets = pd.concat(target_frames).sort_index()

    existing = [col for col in new_targets.columns if col in df.columns]
    if existing:
        print(f"Overwriting existing columns: {', '.join(existing)}")
        df = df.drop(columns=existing)

    df = pd.concat([df, new_targets], axis=1)

    print(f"Saving patched dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Patch consolidated dataset with 10s/20s forward returns.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET, help="Path to earnings_consolidated_final.csv")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path (defaults to --input)")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="Maximum timestamp mismatch (seconds) allowed when matching future bars.",
    )
    args = parser.parse_args()

    output_path = args.output or args.input
    patch_dataset(args.input, output_path, args.tolerance)


if __name__ == "__main__":
    main()


