import pytest


@pytest.fixture
def service_env(monkeypatch, tmp_path):
    # Provide every required setting so service tests do not depend on a local .env.
    values = {
        "MODEL_REGISTRY_DIR": str(tmp_path),
        "LOG_LEVEL": "INFO",
        "TRAIN_DATA_PATH": "unused.csv",
        "VALIDATION_SIZE": "0.2",
        "MODEL_RANDOM_STATE": "42",
        "MIN_ROC_AUC": "0.65",
        "ACTIVE_MODEL_PATH": "",
        "MODEL_C": "10",
        "MODEL_SOLVER": "liblinear",
        "MODEL_MAX_ITER": "1000",
        "MODEL_L1_RATIO": "1.0",
        "ACTIVATION_MAX_ATTEMPTS": "3",
        "ACTIVATION_RETRY_DELAY_SECONDS": "0",
        "INFERENCE_URL": "http://inference:8000",
        "INFERENCE_TIMEOUT_SECONDS": "1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return tmp_path
