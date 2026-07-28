"""Build the engineered hourly feature set used by all forecasting models (Phase 5+).

Granularity assumption: HOURLY, since the feature-engineering checklist
specifies lag windows in hours (1h, 6h, 24h, 168h) rather than days. If you
later decide to model at daily granularity instead, this same structure
applies to the `daily` table with day-based lag windows.

Leakage note: diff/pct-change features are built entirely from already-lagged
values (never from the current, to-be-predicted row) — see add_diff_features.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
DB_PATH = Path("data/energy.duckdb")

TARGET_COL = "global_active_power"

# The 27 days flagged by 2+ methods in Phase 3's anomaly investigation
# (daily grain). Hardcoded since these came out of manual investigation,
# not a function we want to silently re-run and risk different numbers from.
FLAGGED_ANOMALY_DATES = pd.to_datetime([
    "2008-11-23", "2007-02-03", "2007-03-31", "2009-01-18", "2008-11-05",
    "2006-12-16", "2010-09-25", "2010-02-21", "2010-01-14", "2009-12-24",
    "2009-10-17", "2009-09-12", "2009-01-31", "2009-01-17", "2008-12-27",
    "2008-10-26", "2006-12-23", "2008-02-26", "2007-12-01", "2007-10-28",
    "2007-06-08", "2007-03-11", "2007-03-05", "2007-02-18", "2007-01-21",
    "2006-12-26", "2010-10-18",
]).normalize()

LAG_HOURS = [1, 6, 24, 168]
ROLLING_WINDOWS = [24, 168]


def load_hourly() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM hourly ORDER BY datetime").df()
    con.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-based features. These are always known ahead of time (no leakage risk)."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek  # 0 = Monday
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    }
    df["season"] = df["month"].map(season_map)
    return df


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Sin/cos pairs so models see hour 23 and hour 0 as adjacent, not far apart."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """Past values at fixed horizons. shift(N) always looks backward, so these are leak-free."""
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """Rolling mean/std/min/max over past windows.

    shift(1) is applied BEFORE rolling so the window only covers hours
    strictly before the current row. Without this, the rolling window would
    include the current row's own (to-be-predicted) value — a subtle but
    common leakage bug.
    """
    df = df.copy()
    shifted = df[col].shift(1)
    for window in ROLLING_WINDOWS:
        df[f"{col}_roll_mean_{window}h"] = shifted.rolling(window).mean()
        df[f"{col}_roll_std_{window}h"] = shifted.rolling(window).std()
        df[f"{col}_roll_min_{window}h"] = shifted.rolling(window).min()
        df[f"{col}_roll_max_{window}h"] = shifted.rolling(window).max()
    return df


def add_diff_features(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """Momentum features built ONLY from already-lagged values (t-1, t-2, t-25),
    never from the current row — the current row is the forecasting target
    and must not leak into its own input features.
    """
    df = df.copy()
    lag_1 = df[col].shift(1)
    lag_2 = df[col].shift(2)
    lag_25 = df[col].shift(25)  # one hour further back than the 24h lag point

    df[f"{col}_diff_1h"] = lag_1 - lag_2
    df[f"{col}_diff_24h"] = lag_1 - lag_25
    df[f"{col}_pct_change_1h"] = (lag_1 - lag_2) / lag_2.replace(0, np.nan)
    df[f"{col}_pct_change_24h"] = (lag_1 - lag_25) / lag_25.replace(0, np.nan)
    return df


def add_anomaly_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast the Phase 3 daily-level anomaly flags down to every hour of that day."""
    df = df.copy()
    day = pd.Series(df.index.normalize(), index=df.index)
    df["is_flagged_anomaly"] = day.isin(FLAGGED_ANOMALY_DATES).astype(int)
    return df


def build_feature_set() -> pd.DataFrame:
    df = load_hourly()
    df = add_time_features(df)
    df = add_cyclical_encoding(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_diff_features(df)
    df = add_anomaly_flag(df)
    return df


def save_feature_set() -> None:
    df = build_feature_set()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "hourly_features.parquet"
    df.reset_index().to_parquet(out_path, index=False)
    print(f"Saved feature set: {df.shape[0]:,} rows x {df.shape[1]} columns -> {out_path}")

    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS hourly_features")
    con.execute(f"CREATE TABLE hourly_features AS SELECT * FROM read_parquet('{out_path}')")
    con.close()
    print("Loaded into DuckDB table 'hourly_features'.")


if __name__ == "__main__":
    save_feature_set()