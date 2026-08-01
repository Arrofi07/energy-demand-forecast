"""Shared fixtures for the pipeline test suite."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.models.lightgbm_direct import CATEGORICAL_COLS, prepare_features
from src.pipeline.forecaster import ModelBundle


@pytest.fixture
def feature_df():
    """Same shape as tests/models/test_lightgbm_direct.py's fixture: a target
    column plus a couple of engineered predictor columns and the raw
    calendar columns `prepare_features` is expected to exclude.
    """
    idx = pd.date_range("2020-01-01", periods=24 * 20, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "global_active_power": rng.normal(2.0, 0.3, size=len(idx)),
        "hour": idx.hour,
        "dayofweek": idx.dayofweek,
        "month": idx.month,
        "season": "winter",
        "quarter": idx.quarter,
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "is_flagged_anomaly": 0,
        "global_reactive_power": rng.normal(0, 1, size=len(idx)),
        "voltage": 230.0,
        "global_intensity": 5.0,
        "sub_metering_1": 0.0,
        "sub_metering_2": 0.0,
        "sub_metering_3": 0.0,
        "global_active_power_lag_1h": rng.normal(2.0, 0.3, size=len(idx)),
    }, index=idx)


@pytest.fixture
def make_tiny_bundle(feature_df):
    """Factory fixture: `make_tiny_bundle(horizons=(1, 2))` fits a tiny,
    fast LightGBM model per horizon on `feature_df` and returns a
    `ModelBundle` -- everything the pipeline tests need, without waiting on
    a real 24-model, Optuna-tuned fit.
    """

    def _make(horizons=(1, 2)):
        models = {}
        feature_cols = None
        for h in horizons:
            X, y, feature_cols = prepare_features(feature_df, h)
            mask = y.notna() & X.notna().all(axis=1)
            model = lgb.LGBMRegressor(n_estimators=3, num_leaves=7, min_child_samples=1, verbosity=-1)
            model.fit(X.loc[mask], y.loc[mask], categorical_feature=CATEGORICAL_COLS)
            models[h] = model
        return ModelBundle(
            models=models,
            horizons=list(horizons),
            feature_cols=feature_cols,
            trained_through=feature_df.index[-1],
            lgb_params={"n_estimators": 3},
        )

    return _make
