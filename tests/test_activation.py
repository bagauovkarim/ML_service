from unittest.mock import Mock
import requests
from pathlib import Path
from bank_marketing_service import train



def test_activation_result(monkeypatch):
    monkeypatch.setenv("INFERENCE_URL", "http://inference:8000")
    monkeypatch.setenv("INFERENCE_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("ACTIVATION_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("ACTIVATION_RETRY_DELAY_SECONDS", "0")

    attempts = []

    def fake_post(url, json, timeout):
        attempts.append(json)

        if len(attempts) < 3:
            raise requests.exceptions.ConnectionError("temporary error")

        return Mock()

    monkeypatch.setattr(train.requests, "post", fake_post)
    train.activate_inference(Path("model_test.joblib"))

    assert attempts == [{"model_name": "model_test.joblib"}] * 3




def test_activation_raises_after_all_attempts_fail(monkeypatch):
    import pytest

    monkeypatch.setenv("INFERENCE_URL", "http://inference:8000")
    monkeypatch.setenv("INFERENCE_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("ACTIVATION_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("ACTIVATION_RETRY_DELAY_SECONDS", "0")
    failed_post = Mock(side_effect=requests.exceptions.Timeout("unavailable"))
    monkeypatch.setattr(train.requests, "post", failed_post)

    with pytest.raises(requests.exceptions.Timeout):
        train.activate_inference(Path("model_test.joblib"))

    assert failed_post.call_count == 3
