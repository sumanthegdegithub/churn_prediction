import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from churn_model import __version__ as _version
from churn_model.config.core import config
from churn_model.processing.data_manager import load_pipeline
from churn_model.processing.data_manager import pre_pipeline_preparation
from churn_model.processing.validation import validate_inputs
import mlflow.pyfunc
from mlflow import MlflowClient


client = MlflowClient(tracking_uri=config.optuna_configuration.mlflow_tracking_uri)
mlflow.set_tracking_uri(config.optuna_configuration.mlflow_tracking_uri)

champion_model = mlflow.pyfunc.load_model(f"models:/{config.optuna_configuration.project}@{'champion'}")

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    validated_data = validated_data.reindex(columns = config.model_configuration.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
    
    if not errors:
        predictions = champion_model.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}

    return results



if __name__ == "__main__":

    data_in = {'CreditScore': [668], 'Geography': ['France'], 'Gender': ['Male'], 'Age': [33.0], 'Tenure': [3],
               'Balance': [0.0], 'NumOfProducts': [2], 'HasCrCard': [1], 'IsActiveMember': [0]}

    make_prediction(input_data = data_in)