"""Persist and reload backtest results for model comparison.

This module stores evaluation results as Parquet files so that
later analysis can compare forecasting models without rerunning
every experiment.
"""

from pathlib import Path

import pandas as pd


# Directory used to store evaluation results from each forecasting model.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def save_results(df: pd.DataFrame, name: str) -> None:
    """Save a model's evaluation results as a Parquet file."""

    # Create the results directory if it does not already exist.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use the model name as the output filename.
    path = RESULTS_DIR / f"{name}.parquet"

    # Store results in Parquet format for efficient disk usage and fast loading.
    df.to_parquet(path, index=False)

    print(f"Saved {len(df):,} rows -> {path}")


def load_results(name: str) -> pd.DataFrame:
    """Load the saved evaluation results for a single model."""

    return pd.read_parquet(RESULTS_DIR / f"{name}.parquet")


def load_all_results() -> pd.DataFrame:
    """Load and combine evaluation results from all saved models."""

    # Read every Parquet file in the results directory.
    frames = [
        pd.read_parquet(path)
        for path in RESULTS_DIR.glob("*.parquet")
    ]

    # Merge all model results into a single DataFrame
    # for benchmarking and comparison.
    return pd.concat(frames, ignore_index=True)