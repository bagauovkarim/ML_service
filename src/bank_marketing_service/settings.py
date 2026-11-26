from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_registry_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    train_data_path: Path
    validation_size: float = Field(gt=0, lt=1)
    model_random_state: int
    min_roc_auc: float = Field(ge=0, le=1)
    active_model_path: str
    model_c: float = Field(gt=0)
    model_solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    model_max_iter: int = Field(ge=1)
    model_l1_ratio: float = Field(ge=0, le=1)
    activation_max_attempts: int = Field(ge=1)
    activation_retry_delay_seconds: float = Field(ge=0)
    inference_url: HttpUrl
    inference_timeout_seconds: float = Field(gt=0)
