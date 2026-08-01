"""FastAPI wrapper around the Phase 11 production pipeline.

Deliberately thin: every piece of actual forecasting logic (feature
generation, model serialization, prediction) already exists in
`src/pipeline/`. This module's only job is to load the model bundle once at
startup and translate HTTP requests into calls against that existing code --
no forecasting logic is duplicated here.

`get_bundle` / `get_forecaster` are FastAPI dependencies rather than plain
module state so tests can swap in a tiny in-memory bundle via
`app.dependency_overrides` instead of needing the real ~47MB production
bundle on disk.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.schemas import ForecastPoint, ForecastRequest, ForecastResponse, HealthResponse
from src.pipeline.forecaster import DEFAULT_BUNDLE_PATH, DirectMultiHorizonForecaster, ModelBundle, load_bundle
from src.pipeline.inference_features import get_latest_features
from src.pipeline.predict import get_latest_available_origin


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model bundle once at startup rather than per-request --
    it's a 24-model LightGBM bundle, not something to deserialize on every
    call. Fails fast (won't accept traffic) if the bundle doesn't exist,
    e.g. `uv run python -m src.pipeline.registry` was never run.
    """
    app.state.bundle = load_bundle(DEFAULT_BUNDLE_PATH)
    yield
    app.state.bundle = None


app = FastAPI(
    title="Energy Demand Forecast API",
    description="Serves the Phase 7 direct multi-horizon LightGBM model trained in src/pipeline/registry.py.",
    lifespan=lifespan,
)


def get_bundle(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")
    return bundle


def get_forecaster(bundle: ModelBundle = Depends(get_bundle)) -> DirectMultiHorizonForecaster:
    return DirectMultiHorizonForecaster(bundle)


@app.get("/health", response_model=HealthResponse)
def health(bundle: ModelBundle = Depends(get_bundle)) -> HealthResponse:
    try:
        dataset_latest_timestamp = get_latest_available_origin()
        status = "ok"
    except Exception:
        dataset_latest_timestamp = None
        status = "degraded"

    return HealthResponse(
        status=status,
        model_trained_through=bundle.trained_through,
        horizons=bundle.horizons,
        dataset_latest_timestamp=dataset_latest_timestamp,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(
    payload: ForecastRequest = ForecastRequest(),
    forecaster: DirectMultiHorizonForecaster = Depends(get_forecaster),
) -> ForecastResponse:
    """24h-ahead forecast for `payload.as_of` (or the latest available
    timestamp in the dataset if omitted). Does not write to
    `data/forecasts/` -- that's `src/pipeline/predict.py`'s batch/scheduled
    concern; a GET-adjacent read like this shouldn't have that side effect.
    """
    as_of = payload.as_of or get_latest_available_origin()

    try:
        features_row = get_latest_features(as_of)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = forecaster.predict(None, features_row)
    points = [
        ForecastPoint(
            horizon_step=int(row.horizon_step),
            target_timestamp=row.origin + pd.Timedelta(hours=int(row.horizon_step)),
            forecast=float(row.forecast),
        )
        for row in result.itertuples()
    ]
    return ForecastResponse(origin=as_of, points=points)
