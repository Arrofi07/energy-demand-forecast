"""Feature generation for inference -- one forecast origin at a time, from a
bounded recent window, rather than rebuilding the entire historical feature
table the way `src/features/build_features.py`'s `save_feature_set()` does.

Every lag/rolling/diff feature in Phase 4's feature set looks back at most
168 hours (the longest window used anywhere), so `LOOKBACK_HOURS` hours of
history immediately before a forecast origin is always enough to reproduce
the exact same feature values the full-history pipeline would compute for
that same timestamp -- this is verified in
`tests/pipeline/test_inference_features.py` by comparing against
`hourly_features` directly, not just assumed.

Known limitation, stated plainly rather than glossed over: `is_flagged_anomaly`
comes from `add_anomaly_flag`, which checks a hardcoded list of dates found by
Phase 3's *retrospective* full-history investigation. For any date not on that
list -- which, in a real production deployment, means every future date, since
Phase 3 never analyzed data that didn't exist yet -- this feature is always 0.
A live system would need an online anomaly detector to make this feature
meaningful going forward; here it silently degrades to "assume not anomalous,"
which Phase 4 already found wasn't a strong predictor at short horizons anyway.
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.features.build_features import (
    add_anomaly_flag,
    add_cyclical_encoding,
    add_diff_features,
    add_lag_features,
    add_rolling_features,
    add_time_features,
)

# Anchored to the project root, not the caller's cwd -- see the same note
# in src/pipeline/forecaster.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "energy.duckdb"

# Longest rolling/lag window used anywhere in Phase 4 (global_active_power_roll_*_168h,
# lag_168h) plus one hour of margin, so the last row of the window is guaranteed
# NaN-free as long as the underlying data itself has no gap in that stretch.
LOOKBACK_HOURS = 168 + 24


def load_recent_hourly(as_of: pd.Timestamp, lookback_hours: int = LOOKBACK_HOURS,
                        db_path: Path = DB_PATH) -> pd.DataFrame:
    """Pull just enough recent history from DuckDB to compute features at `as_of`.

    This is the only piece of this module that touches the database --
    `build_latest_features` below is a pure function of a DataFrame, so it
    can be tested and reused without a live DuckDB connection.
    """
    window_start = as_of - pd.Timedelta(hours=lookback_hours)
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        "SELECT * FROM hourly WHERE datetime BETWEEN ? AND ? ORDER BY datetime",
        [window_start, as_of],
    ).df()
    con.close()

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").asfreq("h")


def build_latest_features(recent_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Apply the exact Phase 4 feature functions to a recent window, then
    return just the single row at `as_of` -- the row a batch forecast job
    would feed into the direct multi-horizon models.

    Reuses `add_time_features` / `add_cyclical_encoding` / `add_lag_features`
    / `add_rolling_features` / `add_diff_features` / `add_anomaly_flag`
    directly from `src/features/build_features.py`, so any future change to
    how a feature is engineered only has to happen in one place.
    """
    if as_of not in recent_df.index:
        raise ValueError(f"as_of={as_of} not found in the supplied recent_df window.")

    df = add_time_features(recent_df)
    df = add_cyclical_encoding(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_diff_features(df)
    df = add_anomaly_flag(df)

    row = df.loc[[as_of]]
    if row.isna().any(axis=1).item():
        nan_cols = row.columns[row.isna().any()].tolist()
        raise ValueError(
            f"Feature row at {as_of} has NaNs in {nan_cols} -- the recent "
            "window likely doesn't cover enough history (or touches a data "
            "gap) for every lag/rolling feature to be computable."
        )
    return row


def get_latest_features(as_of: pd.Timestamp, lookback_hours: int = LOOKBACK_HOURS,
                         db_path: Path = DB_PATH) -> pd.DataFrame:
    """Convenience wrapper: fetch recent history from DuckDB and build the
    single-row feature set at `as_of`, ready to feed a forecaster.
    """
    recent_df = load_recent_hourly(as_of, lookback_hours, db_path)
    return build_latest_features(recent_df, as_of)
