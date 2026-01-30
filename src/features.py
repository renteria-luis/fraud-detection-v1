# src/features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AmountFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering for the 'Amount' column.
    Learns a threshold for large transactions and applies logarithmic scaling.
    """
    def __init__(self, amount_col='Amount', quantile_threshold=0.95):
        self.amount_col = amount_col
        self.quantile_threshold = quantile_threshold
        self.large_transaction_val_ = None  # Learned threshold value

    def fit(self, X, y=None):
        # 1. LEARN: Compute the quantile value using ONLY training data
        self.large_transaction_val_ = X[self.amount_col].quantile(self.quantile_threshold)
        return self

    def transform(self, X):
        # Validation: if fit was not called, fail loudly
        if self.large_transaction_val_ is None:
            raise RuntimeError("This instance is not fitted yet. Call .fit() first.")
        
        X = X.copy()
        
        # 2. APPLY: Use the stored threshold (self.large_transaction_val_)
        # Log transform to reduce skewness
        X['amount_log'] = np.log1p(X[self.amount_col])
        
        # Binary flags
        X['is_micro_transaction'] = (X[self.amount_col] < 1).astype(int)
        X['is_large_transaction'] = (X[self.amount_col] > self.large_transaction_val_).astype(int)
        
        # Optional: drop the original column if no longer needed
        # return X.drop(columns=[self.amount_col])
        return X


class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms the 'Time' column (seconds) into cyclic features (hour of day).
    No fitting required because time transformations are deterministic.
    """
    def __init__(self, time_col='Time'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self  # Nothing to learn from time

    def transform(self, X):
        X = X.copy()
        
        # Convert seconds to hour of day (0–23)
        # Assumes 'Time' represents seconds since a reference start.
        # Note: in anonymized datasets the start may be arbitrary,
        # but the 24h cyclic pattern is usually preserved.
        X['hour'] = (X[self.time_col] / 3600) % 24
        
        # Feature: is it night time? (e.g., 22:00 to 06:00)
        X['is_night'] = X['hour'].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)
        
        # CYCLIC ENCODING (crucial for Neural Networks and SVMs, useful for Trees)
        # This allows the model to understand that 23:00 and 00:00 are close.
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        
        return X
