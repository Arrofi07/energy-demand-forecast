import lightgbm as lgb
import pandas as pd
import pytest

from src.models.lightgbm_direct import CATEGORICAL_COLS, prepare_features
from src.pipeline.forecaster import DirectMultiHorizonForecaster, load_bundle, save_bundle


class TestBundleRoundTrip:
    def test_save_then_load_preserves_bundle_contents(self, make_tiny_bundle, tmp_path):
        bundle = make_tiny_bundle()
        path = tmp_path / "bundle.joblib"

        save_bundle(bundle, path)
        loaded = load_bundle(path)

        assert loaded.horizons == bundle.horizons
        assert loaded.feature_cols == bundle.feature_cols
        assert loaded.trained_through == bundle.trained_through
        assert set(loaded.models.keys()) == set(bundle.models.keys())

    def test_loaded_bundle_predicts_identically_to_original(self, feature_df, make_tiny_bundle, tmp_path):
        bundle = make_tiny_bundle()
        path = tmp_path / "bundle.joblib"
        save_bundle(bundle, path)
        loaded = load_bundle(path)

        row = feature_df.loc[[feature_df.index[300]]]
        original_preds = DirectMultiHorizonForecaster(bundle).predict(None, row)
        loaded_preds = DirectMultiHorizonForecaster(loaded).predict(None, row)

        pd.testing.assert_frame_equal(original_preds, loaded_preds)


class TestDirectMultiHorizonForecasterPredict:
    def test_returns_long_format_with_expected_shape(self, feature_df, make_tiny_bundle):
        bundle = make_tiny_bundle(horizons=(1, 2, 3))
        rows = feature_df.loc[[feature_df.index[300], feature_df.index[350]]]

        result = DirectMultiHorizonForecaster(bundle).predict(None, rows)

        assert list(result.columns) == ["origin", "horizon_step", "forecast"]
        assert len(result) == len(rows) * len(bundle.horizons)
        assert set(result["horizon_step"]) == {1, 2, 3}
        assert set(result["origin"]) == set(rows.index)

    def test_predictions_match_calling_the_horizon_model_directly(self, feature_df, make_tiny_bundle):
        bundle = make_tiny_bundle(horizons=(1, 2))
        origin = feature_df.index[300]
        row = feature_df.loc[[origin]]

        result = DirectMultiHorizonForecaster(bundle).predict(None, row)

        for h in bundle.horizons:
            X, _, _ = prepare_features(row, h)
            expected = bundle.models[h].predict(X[bundle.feature_cols])[0]
            actual = result.loc[result["horizon_step"] == h, "forecast"].item()
            assert actual == pytest.approx(expected)


class TestCategoricalHandlingAtInference:
    """A bounded inference window can easily have narrower categorical
    coverage than the full training set saw (e.g. a single-day window only
    ever observes one `is_weekend` value). LightGBM is documented to realign
    prediction-time category codes against the categories seen at
    `.fit()` time regardless of what the prediction data's own dtype
    exposes -- this test verifies that behavior directly rather than just
    trusting it, since a silent code mismatch here would corrupt every
    forecast without raising any error.
    """

    def test_narrow_categorical_window_matches_full_category_set(self, feature_df):
        X, y, _ = prepare_features(feature_df, horizon=1)
        mask = y.notna() & X.notna().all(axis=1)
        model = lgb.LGBMRegressor(n_estimators=5, num_leaves=7, min_child_samples=1, verbosity=-1)
        model.fit(X.loc[mask], y.loc[mask], categorical_feature=CATEGORICAL_COLS)

        # is_weekend has both 0 and 1 across the full training set.
        single_row = X.loc[mask].iloc[[0]].copy()
        assert single_row["is_weekend"].cat.categories.tolist() != [single_row["is_weekend"].iloc[0]]

        # Simulate a narrow window: recast is_weekend so its dtype only knows
        # about the one value actually present in this single row.
        narrow_row = single_row.copy()
        narrow_row["is_weekend"] = narrow_row["is_weekend"].astype(object).astype("category")
        assert len(narrow_row["is_weekend"].cat.categories) == 1

        wide_pred = model.predict(single_row)
        narrow_pred = model.predict(narrow_row)

        assert narrow_pred == pytest.approx(wide_pred)
