"""Analyze missing-value gaps in raw minute-level power consumption data."""

from pathlib import Path

import pandas as pd

from src.data.load import load_raw, MEAN_COLS, SUM_COLS


def analyze_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Identify continuous missing-data periods and calculate their durations."""

    # A row is considered missing if at least one measurement column
    # contains a missing value.
    value_cols = MEAN_COLS + SUM_COLS
    is_missing = df[value_cols].isna().any(axis=1)

    # Assign an ID to each consecutive block of missing/non-missing rows.
    # Changes in the boolean sequence indicate a new gap.
    run_id = (is_missing != is_missing.shift()).cumsum()

    # Keep only missing-data blocks and summarize their time ranges.
    gaps = (
        df.assign(is_missing=is_missing, run_id=run_id)
        .loc[lambda d: d["is_missing"]]
        .groupby("run_id")
        .agg(
            start=("datetime", "first"),
            end=("datetime", "last"),
            rows=("datetime", "size"),
        )
    )

    # Calculate the real duration of each gap.
    # Add one minute because both start and end timestamps are included.
    gaps["duration"] = (
        gaps["end"] - gaps["start"] + pd.Timedelta(minutes=1)
    )

    # Sort gaps from longest to shortest for easier inspection.
    return (
        gaps.sort_values("duration", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    """Run missing-value analysis and print a summary report."""

    # Load the raw dataset before any cleaning or interpolation.
    # This ensures the analysis reflects the original data quality.
    df = load_raw()

    total_rows = len(df)

    # Count rows where at least one measurement value is missing.
    missing_rows = (
        df[MEAN_COLS + SUM_COLS]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(f"Total rows: {total_rows:,}")
    print(
        f"Rows with any missing value: "
        f"{missing_rows:,} ({missing_rows / total_rows:.2%})"
    )

    # Identify continuous missing-data periods.
    gaps = analyze_gaps(df)

    print(f"\nNumber of distinct missing-data gaps: {len(gaps)}")
    print("\nTop 10 longest gaps:")
    print(gaps.head(10).to_string(index=False))

    # Evaluate how many missing values belong to large gaps.
    # These gaps are intentionally not interpolated because long
    # artificial reconstructions may introduce unrealistic patterns.
    over_3h = (
        gaps["duration"] > pd.Timedelta(hours=3)
    ).sum()

    rows_in_long_gaps = (
        gaps.loc[
            gaps["duration"] > pd.Timedelta(hours=3),
            "rows",
        ]
        .sum()
    )

    print(
        f"\nGaps longer than 3h: {over_3h} "
        f"(containing {rows_in_long_gaps:,} rows)"
    )

    print(
        f"→ {rows_in_long_gaps / missing_rows:.1%} of all missing rows "
        "will be left as NaN "
        "(not interpolated) under the current 3h threshold."
    )


if __name__ == "__main__":
    main()