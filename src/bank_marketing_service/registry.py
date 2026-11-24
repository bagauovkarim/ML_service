from datetime import datetime, timezone 
from pathlib import Path 
import joblib
from uuid import uuid4
import os

def save_model(pipeline, registry_dir):
    folder = Path(registry_dir)
    folder.mkdir(parents=True, exist_ok=True)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    model_path = folder / f"model_{version}.joblib"

    joblib.dump(pipeline, model_path)
    
    return model_path

def load_model(model_path):
    
    return joblib.load(model_path)

def get_model_path(registry_dir, model_name):
    folder = Path(registry_dir)

    if "/" in model_name or "\\" in model_name or ":" in model_name:
        raise ValueError
    model_path = folder / model_name
    if not model_path.is_file():
        raise FileNotFoundError

    return model_path

def save_active_model_name(registry_dir, model_name):
    get_model_path(registry_dir, model_name)
    path = Path(registry_dir) / "active_model.txt"
    temp_path = Path(registry_dir) / f"active_model_{uuid4().hex}.tmp"
    temp_path.write_text(model_name, encoding="utf-8")
    temp_path.replace(path)

def read_active_model_name(registry_dir):
    path = Path(registry_dir) / "active_model.txt"

    if not os.path.isfile(path):
        return None 

    model_name = path.read_text(encoding="utf-8").strip()

    if not model_name:
        return None 
    
    return model_name