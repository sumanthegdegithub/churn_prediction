import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parents[1], file.parents[2]

PACKAGE_ROOT = root


from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

CONFIG_FILE_PATH = parent / "config.yml"

DATASET_DIR = PACKAGE_ROOT / "data"
TRAINED_MODEL_DIR = parent / "trained_models"



class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    unused_fields: List[str]
    onehot_features: List[str]
    min_max_features: List[str]
    
    test_size: float
    random_state: int
    
class OptunaConfig(BaseModel):
    """
    All configuration relevant to optuna
    training.
    """
    repo_owner: str
    repo_name: str
    experiment_name: str
    artifact_path: str
    model_data_version: int
    run_name: str
    project: str
    optimizer_engine: str
    model_family: str
    feature_set_version: int
    max_depth: List[float]
    learning_rate: List[float]
    n_estimators: List[float]
    min_child_weight: List[float]
    gamma: List[float]
    subsample: List[float]
    colsample_bytree: List[float]
    reg_alpha: List[float]
    reg_lambda: List[float]

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_configuration: ModelConfig
    optuna_configuration: OptunaConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
        
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_configuration = ModelConfig(**parsed_config.data),
        optuna_configuration = OptunaConfig(**parsed_config.data)
    )

    return _config


config = create_and_validate_config()