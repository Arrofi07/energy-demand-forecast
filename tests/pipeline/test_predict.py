import pandas as pd

from src.pipeline.forecaster import save_bundle
from src.pipeline.predict import run_batch_forecast


class TestRunBatchForecast:
    def test_produces_one_row_per_horizon_with_injected_features(self, feature_df, make_tiny_bundle, tmp_path):
        bundle = make_tiny_bundle(horizons=(1, 2, 3))
        bundle_path = tmp_path / "bundle.joblib"
        save_bundle(bundle, bundle_path)

        as_of = feature_df.index[300]
        features_row = feature_df.loc[[as_of]]

        result = run_batch_forecast(
            as_of, bundle_path=bundle_path, features_row=features_row,
            save=False,
        )

        assert len(result) == len(bundle.horizons)
        assert (result["origin"] == as_of).all()
        assert set(result["horizon_step"]) == set(bundle.horizons)

    def test_saves_a_parquet_file_named_for_the_origin(self, feature_df, make_tiny_bundle, tmp_path):
        bundle = make_tiny_bundle(horizons=(1,))
        bundle_path = tmp_path / "bundle.joblib"
        save_bundle(bundle, bundle_path)

        as_of = feature_df.index[300]
        output_dir = tmp_path / "forecasts"

        run_batch_forecast(
            as_of, bundle_path=bundle_path, features_row=feature_df.loc[[as_of]],
            save=True, output_dir=output_dir,
        )

        expected_path = output_dir / f"{as_of:%Y-%m-%dT%H%M}.parquet"
        assert expected_path.exists()
        saved = pd.read_parquet(expected_path)
        assert len(saved) == 1
