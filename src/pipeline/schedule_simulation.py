"""Simulates running the batch forecast pipeline once per day across a date
range -- the same forecast-origin cadence Phases 5-9 used for backtesting,
but walked through the real production code path (DuckDB -> bounded-window
inference features -> serialized model -> forecast) instead of the
notebook's in-memory `direct_multi_horizon_backtest`.

This is a genuine end-to-end validation, not just a demo: if the resulting
MAE/RMSE over the historical test period doesn't match
`06_model_prophet_lightgbm.ipynb`'s LightGBM numbers, that's a real
discrepancy between the research and production code paths worth
investigating -- exactly the kind of gap this simulation exists to catch
before it ever matters on genuinely unseen data.
"""

from pathlib import Path

import pandas as pd

from src.pipeline.forecaster import DEFAULT_BUNDLE_PATH, DirectMultiHorizonForecaster, load_bundle
from src.pipeline.inference_features import get_latest_features

# Anchored to the project root, not the caller's cwd -- see the same note
# in src/pipeline/forecaster.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FORECASTS_DIR = PROJECT_ROOT / "data" / "forecasts"


def simulate_daily_schedule(
    origins: pd.DatetimeIndex,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
) -> pd.DataFrame:
    """Run the batch pipeline once per origin, as if triggered daily by a
    scheduler, and return every forecast concatenated into one long-format
    table. Origins whose recent window can't produce a full, NaN-free
    feature row (e.g. too close to a known data gap) are skipped and
    reported, rather than silently producing a partial/garbage forecast.
    """
    bundle = load_bundle(bundle_path)
    forecaster = DirectMultiHorizonForecaster(bundle)

    all_forecasts = []
    skipped = []
    for origin in origins:
        try:
            features_row = get_latest_features(origin)
        except ValueError:
            skipped.append(origin)
            continue
        all_forecasts.append(forecaster.predict(None, features_row))

    if skipped:
        print(f"Skipped {len(skipped)}/{len(origins)} origins (NaN features near a data gap).")

    if not all_forecasts:
        return pd.DataFrame(columns=["origin", "horizon_step", "forecast"])
    return pd.concat(all_forecasts, ignore_index=True)


def save_simulation(df: pd.DataFrame, name: str = "simulated_schedule",
                     output_dir: Path = FORECASTS_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")
    return path
