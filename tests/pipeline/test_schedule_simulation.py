import pandas as pd
import pytest

from src.pipeline.forecaster import save_bundle
from src.pipeline.schedule_simulation import simulate_daily_schedule


@pytest.fixture
def patched_bundle(monkeypatch, make_tiny_bundle, tmp_path):
    """Saves a tiny bundle to disk and points `load_bundle` (as imported
    into `schedule_simulation`) at it -- the module always loads a bundle by
    path, so patching the constructor it calls is enough to inject the fixture.
    """
    bundle = make_tiny_bundle(horizons=(1, 2))
    path = tmp_path / "bundle.joblib"
    save_bundle(bundle, path)
    return bundle, path


class TestSimulateDailySchedule:
    def test_produces_one_forecast_row_per_origin_per_horizon(self, feature_df, patched_bundle, monkeypatch):
        bundle, bundle_path = patched_bundle
        origins = pd.DatetimeIndex([feature_df.index[100], feature_df.index[200], feature_df.index[300]])
        monkeypatch.setattr(
            "src.pipeline.schedule_simulation.get_latest_features",
            lambda origin: feature_df.loc[[origin]],
        )

        result = simulate_daily_schedule(origins, bundle_path=bundle_path)

        assert len(result) == len(origins) * len(bundle.horizons)
        assert set(result["origin"]) == set(origins)

    def test_skips_origins_whose_features_raise_and_keeps_the_rest(self, feature_df, patched_bundle, monkeypatch, capsys):
        bundle, bundle_path = patched_bundle
        origins = pd.DatetimeIndex([feature_df.index[100], feature_df.index[200], feature_df.index[300]])
        bad_origin = origins[1]

        def flaky_get_features(origin):
            if origin == bad_origin:
                raise ValueError("simulated missing data near a gap")
            return feature_df.loc[[origin]]

        monkeypatch.setattr("src.pipeline.schedule_simulation.get_latest_features", flaky_get_features)

        result = simulate_daily_schedule(origins, bundle_path=bundle_path)

        assert bad_origin not in set(result["origin"])
        assert len(result) == (len(origins) - 1) * len(bundle.horizons)
        assert "Skipped 1/3" in capsys.readouterr().out

    def test_empty_origins_returns_empty_frame_with_expected_columns(self, patched_bundle):
        _, bundle_path = patched_bundle
        result = simulate_daily_schedule(pd.DatetimeIndex([]), bundle_path=bundle_path)
        assert list(result.columns) == ["origin", "horizon_step", "forecast"]
        assert len(result) == 0
