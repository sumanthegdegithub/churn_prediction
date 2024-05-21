from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from churn_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                "CreditScore": 668, 
                "Geography": "France", 
                "Gender": "Male",
                "Age": 33, 
                "Tenure": 3,
                "Balance": 0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1
                    }
                ]
            }
        }
        

