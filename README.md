# Fraud Detection System - V1

End-to-end fraud detection pipeline using scikit-learn.

## Status
🚧 **In Development - V1 (Sklearn Baseline)**

## V1 Objectives
- Complete preprocessing pipeline with sklearn
- Models: LogisticRegression, RandomForest, XGBoost
- Imbalanced data handling with SMOTE
- Basic API with FastAPI
- Dockerized deployment

## Project Structure
```
fraud-detection-v1/
├── data/
│   ├── raw/          # Original datasets
│   └── processed/    # Processed data
├── notebooks/        # Exploratory analysis
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_final_model.ipynb
├── src/              # Source code
│   ├── data/         # Preprocessing modules
│   ├── models/       # Model implementations
│   └── utils/        # Helper functions
├── api/              # FastAPI application
├── models/           # Trained models (.pkl files)
├── requirements.txt
└── Dockerfile
```

## Setup

### 1. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
```bash
# Kaggle Credit Card Fraud Detection
# Place in data/raw/creditcard.csv
```

## Development Roadmap

### Phase 1: V1 - Sklearn Baseline (Weeks 1-3)
- [x] Project structure
- [ ] EDA and feature engineering
- [ ] Baseline models (LogReg, RF, XGB)
- [ ] Imbalance handling (SMOTE)
- [ ] Hyperparameter tuning
- [ ] Model comparison
- [ ] FastAPI endpoint
- [ ] Docker deployment

### Phase 2: V2 - Deep Learning (Weeks 7-10)
- [ ] PyTorch neural network
- [ ] SHAP explainability
- [ ] Model comparison (sklearn vs PyTorch)

### Phase 3: V3 - Testing & MLflow (Weeks 11-14)
- [ ] Unit tests with pytest
- [ ] Integration tests
- [ ] MLflow experiment tracking

### Phase 4: V4 - Production (Weeks 15-18)
- [ ] CI/CD pipeline
- [ ] Monitoring with Prometheus
- [ ] Drift detection

## Tech Stack

**V1 (Current):**
- Python 3.11
- scikit-learn, XGBoost
- imbalanced-learn (SMOTE)
- FastAPI
- Docker

**Planned (V2+):**
- PyTorch
- SHAP
- pytest
- MLflow

## Results (To be updated)

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Baseline | - | - | - | - |
| XGBoost | - | - | - | - |

## Usage (After API is ready)
```bash
# Run API locally
uvicorn api.main:app --reload

# Docker
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection

# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 100.50, "time": 3600, ...}'
```

## License
MIT

## Author
[Your Name] - [Your GitHub]
