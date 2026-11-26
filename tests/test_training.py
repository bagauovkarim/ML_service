from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from bank_marketing_service import train
from bank_marketing_service.settings import Settings


@pytest.fixture
def training_dependencies(monkeypatch, tmp_path):
    settings = {
        "TRAIN_DATA_PATH": "unused.csv",
        "VALIDATION_SIZE": "0.2",
        "MODEL_RANDOM_STATE": "42",
        "MODEL_C": "10",
        "MODEL_SOLVER": "liblinear",
        "MODEL_MAX_ITER": "1000",
        "MODEL_L1_RATIO": "1.0",
        "MIN_ROC_AUC": "0.65",
        "MODEL_REGISTRY_DIR": str(tmp_path),
        "LOG_LEVEL": "INFO",
        "ACTIVE_MODEL_PATH": "",
        "ACTIVATION_MAX_ATTEMPTS": "3",
        "ACTIVATION_RETRY_DELAY_SECONDS": "0",
        "INFERENCE_URL": "http://inference:8000",
        "INFERENCE_TIMEOUT_SECONDS": "1",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)

    # The last two labels form the validation set; no CSV or model fit is needed.
    features = np.arange(10).reshape(-1, 1)
    labels = np.array([0, 1] * 5)
    monkeypatch.setattr(train, "load_training_data", lambda path: (features, labels))
    monkeypatch.setattr(train, "get_preprocessor_scaled", lambda: object())
    monkeypatch.setattr(train, "LogisticRegression", lambda **kwargs: object())

    pipeline = Mock()
    monkeypatch.setattr(train, "Pipeline", lambda steps: pipeline)
    save_model = Mock(return_value=Path(tmp_path) / "model.joblib")
    activate_inference = Mock()
    monkeypatch.setattr(train, "save_model", save_model)
    monkeypatch.setattr(train, "activate_inference", activate_inference)
    return pipeline, save_model, activate_inference


def test_low_auc_does_not_save_or_activate_model(training_dependencies):
    pipeline, save_model, activate_inference = training_dependencies
    pipeline.predict_proba.return_value = np.array([[0.1, 0.9], [0.9, 0.1]])

    with pytest.raises(SystemExit) as error:
        train.main()

    assert error.value.code == 1
    save_model.assert_not_called()
    activate_inference.assert_not_called()


def test_passing_auc_saves_and_activates_model(training_dependencies):
    pipeline, save_model, activate_inference = training_dependencies
    pipeline.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])

    train.main()

    save_model.assert_called_once()
    assert save_model.call_args.args[0] is pipeline
    activate_inference.assert_called_once()
    model_path, passed_settings = activate_inference.call_args.args
    assert model_path == save_model.return_value
    assert isinstance(passed_settings, Settings)
    assert passed_settings.model_registry_dir == save_model.call_args.args[1]
