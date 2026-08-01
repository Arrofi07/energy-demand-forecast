"""Pydantic request/response models for the forecast API.

Kept separate from `main.py` so the request/response contract can be
imported (e.g. by future clients or by the Phase 13 dashboard) without
pulling in the FastAPI app itself.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    as_of: datetime | None = Field(
        default=None,
        description=(
            "Forecast origin timestamp. Defaults to the latest timestamp "
            "available in the dataset if omitted."
        ),
    )


class ForecastPoint(BaseModel):
    horizon_step: int = Field(description="Hours ahead of the forecast origin (1-24).")
    target_timestamp: datetime = Field(description="origin + horizon_step hours.")
    forecast: float = Field(description="Predicted global active power (kW).")


class ForecastResponse(BaseModel):
    origin: datetime
    points: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    model_trained_through: datetime
    horizons: list[int]
    dataset_latest_timestamp: datetime | None
