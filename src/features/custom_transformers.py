from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Feature engineering for transaction data."""
    
    def __init__(self, amount='Amount', time='Time', quantile=0.95):
        """Init column names and quantile threshold."""
        self.amount = amount
        self.time = time
        self.quantile = quantile
        self.threshold_large_ = None
    
    def fit(self, X, y=None):
        """Compute large transaction threshold."""
        self.threshold_large_ = X[self.amount].quantile(self.quantile)
        return self
    
    def transform(self, X):
        """Add features from amount and time."""

        if self.threshold_large_ is None:
            raise RuntimeError("Instance not trained. Run .fit() first.")
        
        X_copy = X.copy()
        
        X_copy['amount_log'] = np.log1p(X_copy[self.amount])        # log of amount
        X_copy['is_micro_transaction'] = (X_copy[self.amount] < 1).astype(int)  # micro tx
        X_copy['is_large_transaction'] = (X_copy[self.amount] > self.threshold_large_).astype(int)  # large tx
        
        hours = (X_copy[self.time] // 3600) % 24
        X_copy['is_night'] = (hours.between(22, 23) | hours.between(0, 6)).astype(int)  # night tx
        X_copy['hour_sin'] = np.sin(2 * np.pi * hours / 24)     # cyclical sin
        X_copy['hour_cos'] = np.cos(2 * np.pi * hours / 24)     # cyclical cos

        X_copy.drop('Time', axis=1, inplace=True)

        return X_copy
