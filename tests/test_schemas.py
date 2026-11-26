import json
import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from bank_marketing_service.inference import app
from bank_marketing_service.schemas import PredictionRequest


@pytest.fixture
def valid_payload():
    return {
        "age": 40,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "euribor3m": 4.8,
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.9,
        "cons.conf.idx": -42.0,
        "nr.employed": 5195.8,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "poutcome": "nonexistent",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", -500),
        ("age", 16),
        ("age", 121),
        ("campaign", -3),
        ("campaign", 0),
        ("pdays", -1),
        ("previous", -1),
        ("euribor3m", math.nan),
        ("emp.var.rate", math.inf),
        ("cons.conf.idx", -math.inf),
    ],
)
def test_prediction_request_rejects_invalid_numbers(valid_payload, field, value):
    valid_payload[field] = value

    with pytest.raises(ValidationError) as error:
        PredictionRequest.model_validate(valid_payload)

    assert (field,) in {tuple(item["loc"]) for item in error.value.errors()}


def test_prediction_request_accepts_boundary_values(valid_payload):
    valid_payload["age"] = 17
    assert PredictionRequest.model_validate(valid_payload).pdays == 999

    valid_payload["age"] = 120
    assert PredictionRequest.model_validate(valid_payload).campaign == 1


@pytest.mark.parametrize(("field", "value"), [("age", -500), ("campaign", -3)])
def test_predict_endpoint_returns_422_for_invalid_numbers(service_env, valid_payload, field, value):
    valid_payload[field] = value

    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert any(item["loc"][-1] == field for item in response.json()["detail"])


def test_predict_endpoint_returns_422_for_nan(service_env, valid_payload):
    valid_payload["euribor3m"] = math.nan

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=json.dumps(valid_payload, allow_nan=True),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 422
    assert any(item["loc"][-1] == "euribor3m" for item in response.json()["detail"])
