import logging
import pickle
import json
from datetime import datetime
from src.config import PAYSIM_PATH, ROOT
from src.data.loader import load_paysim, filter_and_clean
from src.data.splitter import split_data
from src.models.builder import build_pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

THRESHOLD = 0.2226  # from notebook analysis — update if retuned


def train(model_name: str = "xgb", params: dict = None):
    log.info("Loading data...")
    df = load_paysim(PAYSIM_PATH)

    log.info("Filtering and cleaning...")
    X, y = filter_and_clean(df)

    log.info("Splitting...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.15)
    log.info(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    log.info(f"Training {model_name}...")
    pipeline = build_pipeline(model_name, params=params)
    pipeline.fit(X_train, y_train)

    # Metrics
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    y_pred = (y_prob >= THRESHOLD).astype(int)
    precision = (y_pred & y_test).sum() / y_pred.sum()
    recall    = (y_pred & y_test).sum() / y_test.sum()
    f1        = 2 * precision * recall / (precision + recall)

    # Save model
    out = ROOT / "models" / f"fraud_detection_v1_{model_name}.pkl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(pipeline, f)

    # Save metadata
    metadata = {
        'version': '1.0.0',
        'model_type': f'XGBoost (Default)',
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'best_params': 'default',
        'performance': {
            'pr_auc': round(float(pr_auc), 4),
            'production_config': {
                'threshold': float(THRESHOLD),
                'precision': round(float(precision), 4),
                'recall':    round(float(recall), 4),
                'f1':        round(float(f1), 4),
            }
        }
    }
    with open(ROOT / "models" / "metadata_v1.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Model saved to {out}")
    log.info(f"PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    return pipeline, X_test, y_test


if __name__ == "__main__":
    train(model_name="xgb")