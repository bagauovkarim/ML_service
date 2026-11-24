import pytest

from bank_marketing_service.registry import (
    get_model_path,
    read_active_model_name,
    save_active_model_name,
)


def test_active_model_pointer_changes_only_to_existing_file(tmp_path):
    assert read_active_model_name(tmp_path) is None
    (tmp_path / "old.joblib").write_bytes(b"old model")
    (tmp_path / "new.joblib").write_bytes(b"new model")

    save_active_model_name(tmp_path, "old.joblib")
    assert read_active_model_name(tmp_path) == "old.joblib"

    with pytest.raises(FileNotFoundError):
        save_active_model_name(tmp_path, "missing.joblib")
    assert read_active_model_name(tmp_path) == "old.joblib"

    save_active_model_name(tmp_path, "new.joblib")
    assert read_active_model_name(tmp_path) == "new.joblib"
    assert not list(tmp_path.glob("*.tmp"))


def test_registry_rejects_path_outside_its_directory(tmp_path):
    with pytest.raises(ValueError):
        get_model_path(tmp_path, "../other.joblib")
