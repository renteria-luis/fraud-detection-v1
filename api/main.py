import joblib
from pathlib import Path
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException
from api.schemas import FraudApplication, FraudPrediction, HealthCheck
from src.features import FeatureEngineering

# paths
BASE_DIR = Path(__file__).parent.parent  # root of the project
MODELS_DIR = BASE_DIR / 'models'
PARAMS_PATH = BASE_DIR / 'params.yaml'

app = FastAPI(
    title='Fraud Detection API - V1',
    description='Real-time transaction scoring using Tuned XGBoost',
    version='1.0'
)

# singleton for model to avoid reloading on each request
class ModelServer:
    def __init__(self):
        self.model = None
        self.params = None
        self.threshold = 0.5 # default threshold

    def load(self):
        try:
            with open(PARAMS_PATH, 'r') as f:
                config = yaml.safe_load(f)
                # according to ../params.yaml structure:
                self.params = config['v1_xgboost']['params']
                self.threshold = config['v1_xgboost']['deployment']['threshold']

            self.model = joblib.load(MODELS_DIR / 'fraud_detection_v1_xgb.pkl')
            print(f'Model loaded successfully. Operational Threshold: {self.threshold}')
        except Exception as e:
            print(f'Error loading artifacts: {e}')
            raise

server = ModelServer()

@app.on_event('startup')
def startup_event():
    server.load()

@app.get('/health', response_model=HealthCheck)
def health():
    return {
        'status': 'ok',
        'is_model_loaded': server.model is not None,
        'version': '1.0.0'
    }

@app.post('/predict', response_model=FraudPrediction)
def predict(transaction: FraudApplication):
    if server.model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    try:
        input_data = pd.DataFrame([transaction.model_dump()])
        y_prob = server.model.predict_proba(input_data)[0, 1]
    
        return {
            'fraud_probability': float(y_prob),
            'is_fraud': bool(y_prob >= server.threshold),
            'threshold_used': server.threshold,
            'version': '1.0.0'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
