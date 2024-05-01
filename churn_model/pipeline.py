from config.core import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
import pandas as pd
from xgboost import XGBClassifier


class model_pipeline():
    
    def __init__(self, params: dict):

        if not isinstance(params, dict):
            raise ValueError("variable name should be a dictionary")

        self.params = params
        
        preprocesser = ColumnTransformer([
            ('onehotencoder', OneHotEncoder(), config.model_configuration.onehot_features),
            ('minmaxscaler', MinMaxScaler(), config.model_configuration.min_max_features)
        ])
        
        self.pipe = Pipeline([
            ('preprocessor', preprocesser),
            ('model_rf', XGBClassifier(**self.params))
        ])
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # we need the fit statement to accomodate the sklearn pipeline 
        
        self.pipe.fit(X, y)

    
    def predict(self, X: pd.DataFrame):
        
        return self.pipe.predict(X)
    