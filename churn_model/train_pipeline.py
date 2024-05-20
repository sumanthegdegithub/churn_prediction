import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import accuracy_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config.core import PACKAGE_ROOT, config
from pipeline import model_pipeline
from processing.data_manager import load_dataset, pre_pipeline_preparation
import mlflow.pyfunc
from mlflow import MlflowClient

client = MlflowClient(tracking_uri=config.optuna_configuration.mlflow_tracking_uri)
mlflow.set_tracking_uri(config.optuna_configuration.mlflow_tracking_uri)

import dagshub

champion_model = mlflow.pyfunc.load_model(f"models:/{config.optuna_configuration.project}@{'champion'}")


dagshub.init(repo_owner=config.optuna_configuration.repo_owner, repo_name=config.optuna_configuration.repo_name, mlflow=True)

def get_or_create_experiment(experiment_name):

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
experiment_id = get_or_create_experiment(config.optuna_configuration.experiment_name)
mlflow.set_experiment(experiment_id=experiment_id)

# read training data
data = pre_pipeline_preparation(load_dataset(file_name = config.app_config.training_data_file))
test_data = pre_pipeline_preparation(load_dataset(file_name = config.app_config.testing_data_file))

X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_configuration.features],     # predictors
        data[config.model_configuration.target],       # target
        test_size = config.model_configuration.test_size,
        random_state=config.model_configuration.random_state,   # set the random seed here for reproducibility
    )

def objective(trial):
    with mlflow.start_run(nested=True):
        params = {
            'max_depth': trial.suggest_int('max_depth', config.optuna_configuration.max_depth[0], 
                                           config.optuna_configuration.max_depth[1]),
            'learning_rate': trial.suggest_loguniform('learning_rate', config.optuna_configuration.learning_rate[0], 
                                           config.optuna_configuration.learning_rate[1]),
            'n_estimators': trial.suggest_int('n_estimators', config.optuna_configuration.n_estimators[0], 
                                           config.optuna_configuration.n_estimators[1]),
            'min_child_weight': trial.suggest_int('min_child_weight', config.optuna_configuration.min_child_weight[0], 
                                           config.optuna_configuration.min_child_weight[1]),
            'gamma': trial.suggest_loguniform('gamma', config.optuna_configuration.gamma[0], 
                                           config.optuna_configuration.gamma[1]),
            'subsample': trial.suggest_loguniform('subsample', config.optuna_configuration.subsample[0], 
                                           config.optuna_configuration.subsample[1]),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', config.optuna_configuration.colsample_bytree[0], 
                                           config.optuna_configuration.colsample_bytree[1]),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', config.optuna_configuration.reg_alpha[0], 
                                           config.optuna_configuration.reg_alpha[1]),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', config.optuna_configuration.reg_lambda[0], 
                                           config.optuna_configuration.reg_lambda[1]),
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
    
run_name =config.optuna_configuration.run_name
    
    # Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
    # Initialize the Optuna study
    study = optuna.create_study(direction="maximize")

    # Execute the hyperparameter optimization trials.
    # Note the addition of the `champion_callback` inclusion to control our logging
    study.optimize(objective, n_trials=2)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_acc", study.best_value)

    # Log tags
    mlflow.set_tags(
        tags={
            "project": config.optuna_configuration.project,
            "optimizer_engine": config.optuna_configuration.optimizer_engine,
            "model_family": config.optuna_configuration.optimizer_engine,
            "feature_set_version": config.optuna_configuration.feature_set_version,
        }
    )

    pipe = model_pipeline(study.best_params)
    pipe.fit(X_train, y_train)

    # Make predictions
    y_pred = pipe.predict(X_test)

    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    # Log the residuals plot
    artifact_path = config.optuna_configuration.artifact_path

    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path=artifact_path,
        input_example=X_train.iloc[[0]],
        metadata={"model_data_version": config.optuna_configuration.model_data_version},
        registered_model_name = config.optuna_configuration.project,
    )
    
    latest_version = 0
    for mv in client.search_model_versions(f"name='{config.optuna_configuration.project}'"):
        if latest_version < int(dict(mv)['version']):
            latest_version = int(dict(mv)['version'])
            
            
    c_y_pred = champion_model.predict(X_test)
    champion_accuracy = accuracy_score(y_test, c_y_pred)
    print(f'Champion Accuracy {champion_accuracy} challenger Acuracy {accuracy}')
    if accuracy > champion_accuracy: 
        print('Setting challenger as new champion')
        client.set_registered_model_alias(config.optuna_configuration.project, "champion", str(latest_version))
        


