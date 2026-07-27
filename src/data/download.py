"""Download and extract the UCI Individual Household Electric Power Consumption dataset."""

import zipfile
from pathlib import Path

import requests

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "household_power_consumption.zip"
TXT_PATH = RAW_DIR / "household_power_consumption.txt"


def download_dataset(force: bool = False) -> Path:
    """Download the dataset zip if not already present, then extract it."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TXT_PATH.exists() and not force:
        print(f"Raw file already exists at {TXT_PATH}, skipping download.")
        return TXT_PATH

    print(f"Downloading dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)

    if not TXT_PATH.exists():
        raise FileNotFoundError(
            f"Expected {TXT_PATH} after extraction, but it's missing. "
            "Check the archive contents."
        )

    print(f"Done. Raw data at {TXT_PATH}")
    return TXT_PATH


if __name__ == "__main__":
    download_dataset()