from pydantic import BaseModel, Field
from typing import Literal

class FraudApplication(BaseModel):
    step:           int   = Field(..., ge=1, le=744, description='Hour of simulation (1–744)')
    type:           Literal['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'DEBIT']
    amount:         float = Field(..., ge=0, description='Transaction amount')
    nameOrig:       str   = Field(..., description='Origin account ID')
    oldbalanceOrg:  float = Field(..., ge=0, description='Origin balance before transaction')
    nameDest:       str   = Field(..., description='Destination account ID')
    oldbalanceDest: float = Field(..., ge=0, description='Destination balance before transaction')

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 10,
                "type": "TRANSFER",
                "amount": 50000.0,
                "nameOrig": "C123456789",
                "oldbalanceOrg": 50000.0,
                "nameDest": "C987654321",
                "oldbalanceDest": 0.0
            }
        }
    }

class FraudPrediction(BaseModel):
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud:          bool
    threshold_used:    float = Field(ge=0, le=1)
    version:           str

class HealthCheck(BaseModel):
    status:          str
    is_model_loaded: bool
    version:         str