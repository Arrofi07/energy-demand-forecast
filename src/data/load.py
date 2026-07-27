"""Load, clean, validate, and resample the household power consumption dataset."""

from pathlib import Path

import pandas as pd

from src.data.schema import minute_schema

# Input raw dataset
RAW_TXT = Path("data/raw/household_power_consumption.txt")

# Directory where processed datasets will be stored
PROCESSED_DIR = Path("data/processed")

# Rename original dataset columns to snake_case for consistency
COLUMN_RENAME = {
    "Global_active_power": "global_active_power",
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3",
}

# Continuous variables that should be averaged when resampling
MEAN_COLS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
]

# Energy counters that should be summed when resampling
SUM_COLS = [
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

# Only interpolate gaps shorter than 3 hours.
# Longer gaps are preserved as missing values.
MAX_GAP_FOR_INTERPOLATION = pd.Timedelta(hours=3)


def load_raw() -> pd.DataFrame:
    """Load the raw text file and perform basic cleaning."""

    # Read the semicolon-separated dataset and treat '?' as missing values
    df = pd.read_csv(
        RAW_TXT,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
    )

    # Convert column names to snake_case
    df = df.rename(columns=COLUMN_RENAME)

    # Merge Date and Time into a single datetime column
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )

    # Original columns are no longer needed
    df = df.drop(columns=["Date", "Time"])

    # Convert measurement columns to numeric values.
    # Invalid entries become NaN.
    for col in MEAN_COLS + SUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure records are in chronological order
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def interpolate_short_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate gaps up to MAX_GAP_FOR_INTERPOLATION."""

    # Datetime index is required for time-based interpolation
    df = df.set_index("datetime")

    value_cols = MEAN_COLS + SUM_COLS

    # Convert the maximum allowed gap into minutes
    limit = int(MAX_GAP_FOR_INTERPOLATION / pd.Timedelta(minutes=1))

    # Fill only short gaps surrounded by valid observations.
    # Longer gaps remain NaN to avoid introducing unrealistic values.
    df[value_cols] = df[value_cols].interpolate(
        method="time",
        limit=limit,
        limit_area="inside",
    )

    return df.reset_index()


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate the dataset to a new sampling frequency."""

    # Average continuous electrical measurements
    agg = {col: "mean" for col in MEAN_COLS}

    # Sum energy consumption measured by each sub-meter
    agg.update({col: "sum" for col in SUM_COLS})

    # Perform time-based aggregation
    resampled = (
        df.set_index("datetime")
        .resample(freq)
        .agg(agg)
    )

    return resampled.reset_index()


def build_all() -> None:
    """Run the complete preprocessing pipeline."""

    # Create the output directory if it does not exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df = load_raw()

    print("Validating raw schema...")
    minute_schema.validate(df, lazy=True)

    print("Interpolating short gaps...")
    df = interpolate_short_gaps(df)

    print(f"Saving minute-level parquet ({len(df):,} rows)...")
    df.to_parquet(PROCESSED_DIR / "minute.parquet", index=False)

    print("Building hourly aggregate...")
    hourly = resample(df, "h")
    hourly.to_parquet(PROCESSED_DIR / "hourly.parquet", index=False)

    print("Building daily aggregate...")
    daily = resample(df, "D")
    daily.to_parquet(PROCESSED_DIR / "daily.parquet", index=False)

    print("Done.")


if __name__ == "__main__":
    build_all()