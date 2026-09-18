import pytest
from pydantic import ValidationError

from taxi_duration.serving.schemas import PredictionRequest


def test_prediction_contract_accepts_timezone(aware_pickup: object) -> None:
    payload = PredictionRequest(
        pickup_datetime=aware_pickup,
        pickup_location_id=132,
        dropoff_location_id=230,
        passenger_count=1,
    )
    assert payload.pickup_location_id == 132


def test_prediction_contract_rejects_unknown_field(aware_pickup: object) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(
            pickup_datetime=aware_pickup,
            pickup_location_id=132,
            dropoff_location_id=230,
            passenger_count=1,
            fare_amount=60.0,
        )
