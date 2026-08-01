import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_bundle, get_forecaster
from src.pipeline.forecaster import DirectMultiHorizonForecaster


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    """`TestClient(app)` without `with` never runs the app's real lifespan
    (see TestHealth.test_returns_503_when_bundle_not_loaded below), so every
    test here is hermetic regardless of whether the real ~47MB production
    bundle exists on disk -- dependency overrides supply everything a route
    needs directly.
    """
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


class TestHealth:
    def test_returns_503_when_bundle_not_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 503

    def test_returns_ok_with_bundle_metadata_when_loaded(self, client, make_tiny_bundle, monkeypatch):
        bundle = make_tiny_bundle(horizons=(1, 2))
        app.dependency_overrides[get_bundle] = lambda: bundle
        monkeypatch.setattr(
            "src.api.main.get_latest_available_origin",
            lambda: pd.Timestamp("2020-01-20 23:00:00"),
        )

        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["horizons"] == [1, 2]
        assert body["dataset_latest_timestamp"] == "2020-01-20T23:00:00"

    def test_degrades_gracefully_if_dataset_lookup_fails(self, client, make_tiny_bundle, monkeypatch):
        bundle = make_tiny_bundle(horizons=(1,))
        app.dependency_overrides[get_bundle] = lambda: bundle

        def raise_error():
            raise RuntimeError("duckdb file missing")

        monkeypatch.setattr("src.api.main.get_latest_available_origin", raise_error)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"


class TestForecast:
    def test_returns_one_point_per_horizon(self, client, feature_df, make_tiny_bundle, monkeypatch):
        bundle = make_tiny_bundle(horizons=(1, 2, 3))
        app.dependency_overrides[get_forecaster] = lambda: DirectMultiHorizonForecaster(bundle)

        as_of = feature_df.index[300]
        monkeypatch.setattr("src.api.main.get_latest_features", lambda ts: feature_df.loc[[ts]])

        response = client.post("/forecast", json={"as_of": as_of.isoformat()})

        assert response.status_code == 200
        body = response.json()
        assert body["origin"] == as_of.isoformat()
        assert len(body["points"]) == 3
        assert {p["horizon_step"] for p in body["points"]} == {1, 2, 3}
        for point in body["points"]:
            expected_target = (as_of + pd.Timedelta(hours=point["horizon_step"])).isoformat()
            assert point["target_timestamp"] == expected_target

    def test_defaults_as_of_to_latest_available_origin_when_omitted(self, client, feature_df, make_tiny_bundle, monkeypatch):
        bundle = make_tiny_bundle(horizons=(1,))
        app.dependency_overrides[get_forecaster] = lambda: DirectMultiHorizonForecaster(bundle)

        as_of = feature_df.index[300]
        monkeypatch.setattr("src.api.main.get_latest_features", lambda ts: feature_df.loc[[ts]])
        monkeypatch.setattr("src.api.main.get_latest_available_origin", lambda: as_of)

        response = client.post("/forecast", json={})

        assert response.status_code == 200
        assert response.json()["origin"] == as_of.isoformat()

    def test_omitted_body_also_works(self, client, feature_df, make_tiny_bundle, monkeypatch):
        """POST with no body at all should behave identically to `{}` --
        `as_of` is fully optional, not just nullable.
        """
        bundle = make_tiny_bundle(horizons=(1,))
        app.dependency_overrides[get_forecaster] = lambda: DirectMultiHorizonForecaster(bundle)

        as_of = feature_df.index[300]
        monkeypatch.setattr("src.api.main.get_latest_features", lambda ts: feature_df.loc[[ts]])
        monkeypatch.setattr("src.api.main.get_latest_available_origin", lambda: as_of)

        response = client.post("/forecast")

        assert response.status_code == 200
        assert response.json()["origin"] == as_of.isoformat()

    def test_rejects_malformed_as_of_with_422(self, client, make_tiny_bundle):
        # Overriding the forecaster isolates the assertion to body validation
        # specifically -- without it, the missing bundle's 503 would win
        # regardless of what's in the request body.
        bundle = make_tiny_bundle(horizons=(1,))
        app.dependency_overrides[get_forecaster] = lambda: DirectMultiHorizonForecaster(bundle)

        response = client.post("/forecast", json={"as_of": "not-a-date"})

        assert response.status_code == 422

    def test_translates_insufficient_history_value_error_to_400(self, client, make_tiny_bundle, monkeypatch):
        bundle = make_tiny_bundle(horizons=(1,))
        app.dependency_overrides[get_forecaster] = lambda: DirectMultiHorizonForecaster(bundle)

        def raise_value_error(ts):
            raise ValueError("Feature row has NaNs -- insufficient history near a data gap.")

        monkeypatch.setattr("src.api.main.get_latest_features", raise_value_error)

        response = client.post("/forecast", json={"as_of": "2020-01-05T00:00:00"})

        assert response.status_code == 400
        assert "insufficient history" in response.json()["detail"]

    def test_returns_503_when_bundle_not_loaded(self, client):
        response = client.post("/forecast", json={"as_of": "2020-01-05T00:00:00"})
        assert response.status_code == 503
