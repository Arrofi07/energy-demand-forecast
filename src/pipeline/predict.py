"""Batch prediction pipeline: produce one 24h-ahead forecast for a given
forecast origin, saved as Parquet in the same long-format
(`origin`, `horizon_step`, `forecast`) shape used everywhere else in this
project, so a saved batch forecast can be joined against actuals or plugged
into `src/evaluation/baselines.py`'s `compute_metrics` without conversion.
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.pipeline.forecaster import DEFAULT_BUNDLE_PATH, DirectMultiHorizonForecaster, load_bundle
from src.pipeline.inference_features import DB_PATH, get_latest_features

# Anchored to the project root, not the caller's cwd -- see the same note
# in src/pipeline/forecaster.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FORECASTS_DIR = PROJECT_ROOT / "data" / "forecasts"


def get_latest_available_origin(db_path: Path = DB_PATH) -> pd.Timestamp:
    """The most recent timestamp actually present in the `hourly` table.

    This dataset ends in November 2010, so "now" in wall-clock time is
    meaningless as a forecast origin -- a real deployment would use the
    actual current time, but the demo entrypoint below uses this instead so
    it always produces a genuine forecast into hours that don't exist yet in
    the historical data.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    latest = con.execute("SELECT max(datetime) FROM hourly").fetchone()[0]
    con.close()
    return pd.Timestamp(latest)


def run_batch_forecast(
    as_of: pd.Timestamp,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    features_row: pd.DataFrame | None = None,
    save: bool = True,
    output_dir: Path = FORECASTS_DIR,
) -> pd.DataFrame:
    """Produce a 24h-ahead forecast for the single origin `as_of`.

    `features_row` lets a caller (the scheduling simulation, or a test)
    inject a precomputed feature row instead of re-querying DuckDB for every
    origin -- a performance/testability hook only; the forecasting logic
    itself is identical either way.
    """
    if features_row is None:
        features_row = get_latest_features(as_of)

    bundle = load_bundle(bundle_path)
    forecaster = DirectMultiHorizonForecaster(bundle)
    forecast = forecaster.predict(None, features_row)

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{pd.Timestamp(as_of):%Y-%m-%dT%H%M}.parquet"
        forecast.to_parquet(out_path, index=False)
        print(f"Saved {len(forecast)} rows -> {out_path}")

    return forecast


if __name__ == "__main__":
    origin = get_latest_available_origin()
    print(f"Forecasting 24h ahead of the latest available data point: {origin}")
    run_batch_forecast(origin)
