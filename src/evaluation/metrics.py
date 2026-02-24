# src/evaluation/metrics.py
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, auc, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str = 'Model', ax=None, threshold: float = 0.5, plot: bool = True)-> dict:
    """
    Evaluate a binary classification model.
    Agnostic to dataset — works for any sklearn-compatible pipeline.
    Parameters
    ----------
    threshold : float
        Decision threshold for classification report. Default 0.5.
        Tune this after inspecting the PR curve.
    plot : bool
        Whether to render the Precision/Recall vs Threshold chart. Default True.
    """
    # predictions
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    # default 0.5 for baseline metrics
    y_test_pred  = (y_test_prob >= threshold).astype(int)
    # overfitting check (using PR-AUC as it's more sensitive to class imbalance)
    train_prec, train_rec, _ = precision_recall_curve(y_train, y_train_prob)
    train_pr_auc = auc(train_rec, train_prec)
    cm = confusion_matrix(y_test, y_test_pred)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_prob)
    pr_auc   = auc(recalls, precisions)
    test_auc = roc_auc_score(y_test, y_test_prob)
    gap = train_pr_auc - pr_auc
    if gap > 0.10 and plot == True:  # arbitrary threshold for alerting on overfitting
        print(f"ALERT: Overfitting in {model_name} — PR-AUC gap: {gap*100:.2f}%")
    # Plot
    if plot:
        show = ax is None
        if show:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
        ax.plot(thresholds, recalls[:-1],    "g-",  label="Recall")
        ax.set_xlabel("Threshold")
        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.legend(loc="best")
        ax.set_title(f"{model_name} | PR AUC: {pr_auc:.4f} | ROC AUC: {test_auc:.4f}")
        ax.grid(True, linestyle="--", linewidth=0.5)
        if show:
            plt.tight_layout()
            plt.show()
    return {
        'model': model,
        'test_auc': test_auc,
        'pr_auc': pr_auc,
        'train_pr_auc': train_pr_auc,
        'precisions': precisions,
        'recalls': recalls,
        'thresholds': thresholds,
        'report': classification_report(y_test, y_test_pred),
        'confusion_matrix': cm,
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall':    recall_score(y_test, y_test_pred, zero_division=0),
        'f1':        f1_score(y_test, y_test_pred, zero_division=0),
    }