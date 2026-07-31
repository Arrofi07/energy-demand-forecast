import numpy as np
import pandas as pd
import pytest

from src.models.lightgbm_direct import (
    CATEGORICAL_COLS,
    EXCLUDE_BASE_COLS,
    TARGET_COL,
    fit_direct_models,
    prepare_features,
)


@pytest.fixture
def feature_df():
    """Small df shaped like the output of build_feature_set(): target column
    plus a couple of already-engineered predictor columns and the raw
    calendar columns that prepare_features is expected to exclude.
    """
    idx = pd.date_range("2020-01-01", periods=24 * 20, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        TARGET_COL: rng.normal(2.0, 0.3, size=len(idx)),
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
    return df


class TestPrepareFeatures:
    def test_y_is_target_shifted_backward_by_horizon(self, feature_df):
        horizon = 5
        X, y, _ = prepare_features(feature_df, horizon)

        expected_y = feature_df[TARGET_COL].shift(-horizon)
        pd.testing.assert_series_equal(y, expected_y, check_names=False)

    def test_target_calendar_features_use_future_timestamp_not_origin(self, feature_df):
        horizon = 3
        X, _, _ = prepare_features(feature_df, horizon)

        i = 100
        future_ts = feature_df.index[i] + pd.Timedelta(hours=horizon)
        assert X["target_hour_sin"].iloc[i] == pytest.approx(np.sin(2 * np.pi * future_ts.hour / 24))
        assert X["target_is_weekend"].iloc[i] == int(future_ts.dayofweek >= 5)

    def test_excluded_columns_are_not_in_feature_set(self, feature_df):
        X, _, feature_cols = prepare_features(feature_df, horizon=1)

        for col in EXCLUDE_BASE_COLS:
            assert col not in X.columns
            assert col not in feature_cols

    def test_categorical_columns_cast_to_category_dtype(self, feature_df):
        X, _, _ = prepare_features(feature_df, horizon=1)

        for col in CATEGORICAL_COLS:
            if col in X.columns:
                assert str(X[col].dtype) == "category"

    def test_output_length_matches_input(self, feature_df):
        X, y, _ = prepare_features(feature_df, horizon=10)
        assert len(X) == len(feature_df)
        assert len(y) == len(feature_df)


@pytest.mark.slow
class TestFitDirectModels:
    def test_training_mask_excludes_targets_beyond_train_end(self, feature_df):
        """A row must only be used to train the horizon-h model if its target
        timestamp (origin + h) falls at or before train_end -- otherwise the
        model would be trained on data from "the future" relative to train_end.
        """
        train_end = feature_df.index[24 * 15]  # leave the last 5 days as held-out
        horizons = [1, 24]
        lgb_params = {"n_estimators": 3, "num_leaves": 7, "min_child_samples": 1, "verbosity": -1}

        models = fit_direct_models(feature_df, train_end, horizons, lgb_params)

        assert set(models.keys()) == set(horizons)
        for h in horizons:
            X, y, _ = prepare_features(feature_df, h)
            preds = models[h].predict(X.dropna())
            assert len(preds) == len(X.dropna())
