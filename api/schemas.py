from pydantic import BaseModel, Field

class FraudApplication(BaseModel):
    """Input schema for a single transaction."""

    Time: float = Field(..., ge=0, description='Elapsed time recorded in seconds')
    Amount: float = Field(..., ge=0, description='Attempted transfer amount')
    V1: float = Field(..., description='PCA component 1 derived from original transaction features')
    V2: float = Field(..., description='PCA component 2 derived from original transaction features')
    V3: float = Field(..., description='PCA component 3 derived from original transaction features')
    V4: float = Field(..., description='PCA component 4 derived from original transaction features')
    V5: float = Field(..., description='PCA component 5 derived from original transaction features')
    V6: float = Field(..., description='PCA component 6 derived from original transaction features')
    V7: float = Field(..., description='PCA component 7 derived from original transaction features')
    V8: float = Field(..., description='PCA component 8 derived from original transaction features')
    V9: float = Field(..., description='PCA component 9 derived from original transaction features')
    V10: float = Field(..., description='PCA component 10 derived from original transaction features')
    V11: float = Field(..., description='PCA component 11 derived from original transaction features')
    V12: float = Field(..., description='PCA component 12 derived from original transaction features')
    V13: float = Field(..., description='PCA component 13 derived from original transaction features')
    V14: float = Field(..., description='PCA component 14 derived from original transaction features')
    V15: float = Field(..., description='PCA component 15 derived from original transaction features')
    V16: float = Field(..., description='PCA component 16 derived from original transaction features')
    V17: float = Field(..., description='PCA component 17 derived from original transaction features')
    V18: float = Field(..., description='PCA component 18 derived from original transaction features')
    V19: float = Field(..., description='PCA component 19 derived from original transaction features')
    V20: float = Field(..., description='PCA component 20 derived from original transaction features')
    V21: float = Field(..., description='PCA component 21 derived from original transaction features')
    V22: float = Field(..., description='PCA component 22 derived from original transaction features')
    V23: float = Field(..., description='PCA component 23 derived from original transaction features')
    V24: float = Field(..., description='PCA component 24 derived from original transaction features')
    V25: float = Field(..., description='PCA component 25 derived from original transaction features')
    V26: float = Field(..., description='PCA component 26 derived from original transaction features')
    V27: float = Field(..., description='PCA component 27 derived from original transaction features')
    V28: float = Field(..., description='PCA component 28 derived from original transaction features')

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 172788.0,
                "Amount": 67.88,
                "V1": 1.92,
                "V2": -0.301,
                "V3": -3.25,
                "V4": -0.558,
                "V5": 2.631,
                "V6": 3.031,
                "V7": -0.297,
                "V8": 0.708,
                "V9": 0.432,
                "V10": -0.485,
                "V11": 0.412,
                "V12": 0.063,
                "V13": -0.184,
                "V14": -0.511,
                "V15": 1.329,
                "V16": 0.141,
                "V17": 0.314,
                "V18": 0.396,
                "V19": -0.577,
                "V20": 0.001,
                "V21": 0.232,
                "V22": 0.578,
                "V23": -0.038,
                "V24": 0.64,
                "V25": 0.266,
                "V26": -0.087,
                "V27": 0.004,
                "V28": -0.027
            }
        }
    }

class FraudPrediction(BaseModel):
    """Output schema for prediction."""

    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    threshold_used: float = Field(ge=0, le=1)
    model_version: str

class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: str
    