"""Direct multi-horizon LightGBM forecasting for the 24h-ahead benchmark.

This implementation trains 24 independent models (one for each forecast
horizon h=1..24). Each model predicts a specific future timestamp using:

- Historical origin-time features generated in Phase 4 (already checked
  to be leak-free).
- Deterministic calendar features from the target timestamp, such as
  future hour, weekday, month, and weekend indicator.

A direct forecasting strategy is used instead of recursive forecasting to
avoid error accumulation when predicting multiple steps ahead.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb


# Target variable to forecast
TARGET_COL = "global_active_power"


# Features treated as categorical by LightGBM.
# They represent discrete states rather than continuous measurements.
CATEGORICAL_COLS = [
    "quarter",
    "is_weekend",
    "is_flagged_anomaly",
    "target_is_weekend",
]


# Columns excluded from model input:
# - target variable (to prevent leakage)
# - raw calendar columns replaced by encoded versions
# - electrical measurements not used as predictors
EXCLUDE_BASE_COLS = [
    TARGET_COL,
    "hour",
    "dayofweek",
    "month",
    "season",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]


def prepare_features(
    df: pd.DataFrame,
    horizon: int,
):
    """Create training features and labels for one forecast horizon.

    For horizon h:
        X = information available at forecast origin
            + known calendar information at origin + h

        y = observed target value at origin + h

    This transformation ensures that each model learns one specific
    forecasting distance without seeing future measurements.
    """

    data = df.copy()

    # Calculate the timestamp being predicted.
    # Calendar properties of this timestamp are known in advance
    # and therefore do not introduce future data leakage.
    future_index = data.index + pd.Timedelta(hours=horizon)

    # Encode cyclic calendar features.
    # Sine/cosine transformations preserve the circular relationship:
    # hour 23 and hour 0 are close to each other.
    data["target_hour_sin"] = np.sin(
        2 * np.pi * future_index.hour / 24
    )
    data["target_hour_cos"] = np.cos(
        2 * np.pi * future_index.hour / 24
    )

    data["target_dayofweek_sin"] = np.sin(
        2 * np.pi * future_index.dayofweek / 7
    )
    data["target_dayofweek_cos"] = np.cos(
        2 * np.pi * future_index.dayofweek / 7
    )

    data["target_month_sin"] = np.sin(
        2 * np.pi * future_index.month / 12
    )
    data["target_month_cos"] = np.cos(
        2 * np.pi * future_index.month / 12
    )

    # Weekend information for the exact timestamp being predicted.
    data["target_is_weekend"] = (
        future_index.dayofweek >= 5
    ).astype(int)

    # Shift target backwards so each row contains:
    # current features -> future value at horizon h.
    y = data[TARGET_COL].shift(-horizon)

    # Select only valid predictor columns.
    feature_cols = [
        c for c in data.columns
        if c not in EXCLUDE_BASE_COLS
    ]

    X = data[feature_cols].copy()

    # Tell LightGBM which variables should be handled categorically.
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return X, y, feature_cols


def fit_direct_models(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    horizons,
    lgb_params: dict,
) -> dict:
    """Train one independent LightGBM model for each forecast horizon."""

    models = {}

    for h in horizons:

        # Create features and labels specifically for this horizon.
        X, y, _ = prepare_features(df, h)

        # Prevent leakage:
        # A training row is allowed only if its prediction target
        # occurs before or exactly at the end of the training period.
        mask = (
            (df.index <= train_end - pd.Timedelta(hours=h))
            & y.notna()
            & X.notna().all(axis=1)
        )

        # Create and train the horizon-specific model.
        model = lgb.LGBMRegressor(**lgb_params)

        model.fit(
            X.loc[mask],
            y.loc[mask],
            categorical_feature=CATEGORICAL_COLS,
        )

        models[h] = model

    return models


def direct_multi_horizon_backtest(
    df: pd.DataFrame,
    models: dict,
    origins: pd.DatetimeIndex,
    horizon_max: int,
    method_name: str,
) -> pd.DataFrame:
    """Evaluate all direct horizon models on shared forecast origins.

    Predictions are generated horizon-by-horizon rather than
    origin-by-origin to reduce unnecessary model prediction calls.
    """

    records = []

    for h in range(1, horizon_max + 1):

        # Generate the feature matrix for this forecast horizon.
        X, _, _ = prepare_features(df, h)

        # Keep only rows corresponding to the evaluation origins.
        origin_mask = (
            X.index.isin(origins)
            & X.notna().all(axis=1)
        )

        X_origins = X.loc[origin_mask]

        # Predict the h-step-ahead value using the dedicated model.
        preds = models[h].predict(X_origins)

        # Retrieve the actual future observations for comparison.
        actuals = df[TARGET_COL].reindex(
            X_origins.index + pd.Timedelta(hours=h)
        )

        # Store results in the common benchmark format.
        for origin, forecast, actual in zip(
            X_origins.index,
            preds,
            actuals.values,
        ):
            records.append(
                {
                    "origin": origin,
                    "horizon_step": h,
                    "method": method_name,
                    "actual": actual,
                    "forecast": forecast,
                }
            )

    return pd.DataFrame.from_records(records)