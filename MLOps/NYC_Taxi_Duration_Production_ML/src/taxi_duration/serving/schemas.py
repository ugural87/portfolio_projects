from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pickup_datetime: datetime
    # TLC zones 264 (Unknown) and 265 (Outside of NYC) are not routable destinations.
    pickup_location_id: int = Field(ge=1, le=263)
    dropoff_location_id: int = Field(ge=1, le=263)
    passenger_count: int = Field(ge=1, le=6)

    @field_validator("pickup_datetime")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("pickup_datetime must include a UTC offset")
        return value


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float
    model_version: str
    feature_contract_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
