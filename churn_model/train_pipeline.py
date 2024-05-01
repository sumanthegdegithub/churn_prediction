import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import cmd
import textwrap
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
import mlflow
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import optuna

from config.core import PACKAGE_ROOT, config
from pipeline import model_pipeline
from processing.data_manager import load_dataset, save_pipeline, pre_pipeline_preparation


import dagshub
dagshub.init(repo_owner='sumanthegdegithub', repo_name='churn_prediction', mlflow=True)

def get_or_create_experiment(experiment_name):

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
experiment_id = get_or_create_experiment(config.app_config.experiment_name)
mlflow.set_experiment(experiment_id=experiment_id)

# read training data
data = pre_pipeline_preparation(load_dataset(file_name = config.app_config.training_data_file))

X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_configuration.features],     # predictors
        data[config.model_configuration.target],       # target
        test_size = config.model_configuration.test_size,
        random_state=config.model_configuration.random_state,   # set the random seed here for reproducibility
    )

def objective(trial):
    with mlflow.start_run(nested=True):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        }

        pipe = model_pipeline(params)
        pipe.fit(X_train, y_train)

        # Make predictions
        y_pred = pipe.predict(X_test)

        # Evaluate predictions
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        
        return accuracy
    
run_name = 'first_run'
    
    # Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
    # Initialize the Optuna study
    study = optuna.create_study(direction="maximize")

    # Execute the hyperparameter optimization trials.
    # Note the addition of the `champion_callback` inclusion to control our logging
    study.optimize(objective, n_trials=30)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_acc", study.best_value)

    # Log tags
    mlflow.set_tags(
        tags={
            "project": "churn_prediction",
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
            "feature_set_version": 1,
        }
    )

    pipe = model_pipeline(study.best_params)
    pipe.fit(X_train, y_train)

    # Make predictions
    y_pred = pipe.predict(X_test)

    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    # Log the residuals plot

    artifact_path = "model"

    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path=artifact_path,
        input_example=X_train.iloc[[0]],
        metadata={"model_data_version": 1},
    )

    # Get the logged model uri so that we can load it from the artifact store
    model_uri = mlflow.get_artifact_uri(artifact_path)

