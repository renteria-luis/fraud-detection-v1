from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, auc
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name='Model', ax=None):
    """
    Evaluate a binary classification model with ROC AUC, PR AUC, and threshold tuning visualization.
    Returns
    -------
    dict
        Dictionary containing:
        - 'model': trained model
        - 'test_auc': ROC AUC on test set
        - 'pr_auc': PR AUC on test set
        - 'precisions': array of precisions for all thresholds
        - 'recalls': array of recalls for all thresholds
        - 'thresholds': array of thresholds corresponding to precision-recall points
        - 'report': classification report (threshold 0.5)
    """
    # predictions
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # default 0.5 for baseline metrics
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    
    # overfitting check (using PR-AUC as it's more sensitive to class imbalance)
    train_prec, train_rec, _ = precision_recall_curve(y_train, y_train_prob)
    train_pr_auc = auc(train_rec, train_prec)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    # threshold tuning plot (test set)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(recalls, precisions)
    
    # print(f"Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Gap: {train_auc - test_auc:.4f}")
    if (train_pr_auc - pr_auc) > 0.10:  # > 10% diff
        print(f'ALERT: Signs of overfitting detected in {model_name}: Diff: {(train_pr_auc - pr_auc) * 100:.4f}%.')
    
    # print(f"PR AUC: {pr_auc:.4f}")
    
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.legend(loc="best")
    ax.set_title(f"{model_name}: Precision-Recall Tradeoff | PR AUC: {pr_auc:.4f}")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if ax is plt.gca():
        plt.show()
    
    return {
        'model': model,
        'test_auc': test_auc,
        'pr_auc': pr_auc,
        'precisions': precisions,
        'recalls': recalls,
        'thresholds': thresholds,
        'report': classification_report(y_test, y_test_pred)
    }