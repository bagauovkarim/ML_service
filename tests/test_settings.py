from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import HttpUrl, ValidationError

from bank_marketing_service import train
from bank_marketing_service.settings import Settings


def test_settings_load_valid_environment(service_env):
    settings = Settings()

    assert settings.model_registry_dir == service_env
    assert settings.log_level == "INFO"
    assert settings.model_solver == "liblinear"
    assert isinstance(settings.inference_url, HttpUrl)


@pytest.mark.parametrize(
    ("variable", "invalid_value"),
    [
        ("LOG_LEVEL", "nonsense"),
        ("MODEL_SOLVER", "nonsense"),
        ("INFERENCE_URL", "not-a-url"),
    ],
)
def test_settings_reject_invalid_environment(service_env, monkeypatch, variable, invalid_value):
    monkeypatch.setenv(variable, invalid_value)

    with pytest.raises(ValidationError) as error:
        Settings()

    assert (variable.lower(),) in {tuple(item["loc"]) for item in error.value.errors()}


def test_activation_uses_validated_url(service_env, monkeypatch):
    settings = Settings()
    post = Mock()
    monkeypatch.setattr(train.requests, "post", post)

    train.activate_inference(Path("model_test.joblib"), settings)

    post.assert_called_once_with(
        "http://inference:8000/models/activate",
        json={"model_name": "model_test.joblib"},
        timeout=1.0,
    )
