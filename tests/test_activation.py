from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from bank_marketing_service import train


@pytest.fixture
def activation_settings():
    # These tests exercise retries, not environment loading.
    return SimpleNamespace(
        inference_url="http://inference:8000",
        inference_timeout_seconds=1.0,
        activation_max_attempts=3,
        activation_retry_delay_seconds=0.0,
    )


def test_activation_result(monkeypatch, activation_settings):
    attempts = []

    def fake_post(url, json, timeout):
        attempts.append((url, json, timeout))
        if len(attempts) < 3:
            raise requests.exceptions.ConnectionError("temporary error")
        return Mock()

    monkeypatch.setattr(train.requests, "post", fake_post)
    train.activate_inference(Path("model_test.joblib"), activation_settings)

    expected_call = (
        "http://inference:8000/models/activate",
        {"model_name": "model_test.joblib"},
        1.0,
    )
    assert attempts == [expected_call] * 3


def test_activation_raises_after_all_attempts_fail(monkeypatch, activation_settings):
    failed_post = Mock(side_effect=requests.exceptions.Timeout("unavailable"))
    monkeypatch.setattr(train.requests, "post", failed_post)

    with pytest.raises(requests.exceptions.Timeout):
        train.activate_inference(Path("model_test.joblib"), activation_settings)

    assert failed_post.call_count == 3
