from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from bank_marketing_service import inference
from bank_marketing_service.registry import read_active_model_name, save_active_model_name
from bank_marketing_service.schemas import ModelSwitchRequest


def test_invalid_new_model_preserves_active_model(tmp_path):
    (tmp_path / "old.joblib").write_bytes(b"old model")
    (tmp_path / "broken.joblib").write_bytes(b"not a joblib file")
    save_active_model_name(tmp_path, "old.joblib")

    old_model = object()
    state = SimpleNamespace(
        model=old_model,
        model_name="old.joblib",
        settings=SimpleNamespace(model_registry_dir=tmp_path),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    with pytest.raises(HTTPException) as error:
        inference.switch_model(ModelSwitchRequest(model_name="broken.joblib"), request)

    assert error.value.status_code == 500
    assert state.model is old_model
    assert state.model_name == "old.joblib"
    assert read_active_model_name(tmp_path) == "old.joblib"


def test_valid_new_model_updates_pointer_and_running_model(tmp_path):
    import joblib
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    (tmp_path / "old.joblib").write_bytes(b"old model")
    save_active_model_name(tmp_path, "old.joblib")
    new_model = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    new_model.fit(np.array([[0], [1]]), np.array([0, 1]))
    joblib.dump(new_model, tmp_path / "new.joblib")

    old_model = object()
    state = SimpleNamespace(
        model=old_model,
        model_name="old.joblib",
        settings=SimpleNamespace(model_registry_dir=tmp_path),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    response = inference.switch_model(ModelSwitchRequest(model_name="new.joblib"), request)

    assert response == {"active_model": "new.joblib"}
    assert state.model is not old_model
    assert state.model_name == "new.joblib"
    assert read_active_model_name(tmp_path) == "new.joblib"


def test_prediction_uses_shared_preprocessing_and_positive_class_probability():
    import numpy as np
    import pandas as pd
    from unittest.mock import Mock

    from bank_marketing_service.schemas import PredictionRequest

    payload = PredictionRequest.model_validate({
        "age": 40,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "euribor3m": 4.8,
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.9,
        "cons.conf.idx": -42.0,
        "nr.employed": 5195.8,
        "job": "unknown",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "poutcome": "nonexistent",
    })
    model = Mock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model=model)))

    response = inference.predict(payload, request)

    assert response == {"prediction": 1, "probability": 0.8}
    prepared = model.predict.call_args.args[0]
    assert prepared.loc[0, "pdays"] == 0
    assert pd.isna(prepared.loc[0, "job"])
    assert model.predict_proba.call_args.args[0] is prepared


def test_empty_registry_starts_without_model_and_reports_unavailable(service_env):
    import asyncio
    from fastapi import FastAPI

    app = FastAPI()

    async def check_startup():
        async with inference.lifespan(app):
            assert app.state.model is None
            with pytest.raises(HTTPException) as error:
                inference.health(SimpleNamespace(app=app))
            assert error.value.status_code == 503

    asyncio.run(check_startup())


def test_activate_endpoint_accepts_model_name_without_settings_in_json(service_env):
    import joblib
    import numpy as np
    from fastapi.testclient import TestClient
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    model = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    model.fit(np.array([[0], [1]]), np.array([0, 1]))
    joblib.dump(model, service_env / "new.joblib")

    with TestClient(inference.app) as client:
        response = client.post("/models/activate", json={"model_name": "new.joblib"})

    assert response.status_code == 200
    assert response.json() == {"active_model": "new.joblib"}
    assert read_active_model_name(service_env) == "new.joblib"
