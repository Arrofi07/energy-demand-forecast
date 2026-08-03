"""Tests for src/pipeline/registry.py.

Every test here uses a temporary MLflow tracking URI and a temporary/unique
model name -- never the real project's `mlruns/` store -- so running the
test suite never pollutes the actual registered "energy-demand-lightgbm-direct"
model. `train_production_bundle` still trains on the real DuckDB data (there's
no fixture standing in for it), so these hit real data and real LightGBM
fits, hence the `slow` marker.
"""

import mlflow
import pytest

from src.pipeline import registry
from src.pipeline.forecaster import DirectMultiHorizonForecaster
from src.pipeline.inference_features import get_latest_features
from src.pipeline.predict import get_latest_available_origin

TINY_PARAMS = {
    "n_estimators": 3,
    "num_leaves": 7,
    "min_child_samples": 1,
    "objective": "regression",
    "verbosity": -1,
    "random_state": 42,
}


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Points registry.py's tracking URI and model name at a throwaway
    location for the duration of one test.
    """
    unique_name = f"test-model-{tmp_path.name}"
    monkeypatch.setattr(registry, "TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(registry, "MODEL_NAME", unique_name)
    return unique_name


@pytest.mark.slow
class TestTrainProductionBundle:
    def test_trains_all_24_horizons(self):
        bundle = registry.train_production_bundle(lgb_params=TINY_PARAMS)

        assert set(bundle.models.keys()) == set(range(1, 25))
        assert bundle.lgb_params == TINY_PARAMS

    def test_bundle_produces_a_working_forecaster_on_real_data(self):
        bundle = registry.train_production_bundle(lgb_params=TINY_PARAMS)
        origin = get_latest_available_origin()
        features_row = get_latest_features(origin)

        result = DirectMultiHorizonForecaster(bundle).predict(None, features_row)

        assert len(result) == 24
        assert result["forecast"].notna().all()


@pytest.mark.slow
class TestRegisterBundle:
    def test_registers_a_version_reachable_by_the_champion_alias(self, isolated_mlflow):
        bundle = registry.train_production_bundle(lgb_params=TINY_PARAMS)

        model_uri = registry.register_bundle(bundle, alias="champion")

        assert model_uri == f"models:/{isolated_mlflow}@champion"
        client = mlflow.tracking.MlflowClient(tracking_uri=registry.TRACKING_URI)
        version = client.get_model_version_by_alias(isolated_mlflow, "champion")
        assert version is not None

    def test_repeated_registration_creates_a_new_version_and_moves_the_alias(self, isolated_mlflow):
        bundle = registry.train_production_bundle(lgb_params=TINY_PARAMS)

        first_uri = registry.register_bundle(bundle, alias="champion")
        second_uri = registry.register_bundle(bundle, alias="champion")

        assert first_uri == second_uri  # alias-based URI string is stable...
        client = mlflow.tracking.MlflowClient(tracking_uri=registry.TRACKING_URI)
        versions = client.search_model_versions(f"name='{isolated_mlflow}'")
        assert len(versions) == 2  # ...but it now points at a 2nd version
        champion = client.get_model_version_by_alias(isolated_mlflow, "champion")
        assert int(champion.version) == 2
