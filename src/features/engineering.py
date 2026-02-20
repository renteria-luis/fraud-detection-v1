# src/features/engineering.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class PaySimFeatures(BaseEstimator, TransformerMixin):
    '''
    Feature engineering for PaySim dataset.
    All features use only pre-transaction state (no leakage).
    Expects newbalanceOrig and newbalanceDest already dropped.
    '''

    def __init__(self, cyclical_encoding: bool = False):
        # cyclical_encoding=True for logreg/DL, False for tree-based
        self.cyclical_encoding = cyclical_encoding

    def fit(self, X, y=None):
        return self  # stateless, no statistics to learn

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # time features
        hour = (X['step'] % 24).astype('int8')
        X['hour_of_day'] = hour
        X['day'] = (X['step'] // 24).astype('int16')
        X = X.drop(columns=['step'])

        # cyclical
        if self.cyclical_encoding:
            X['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype('float32')
            X['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype('float32')

        # amount
        X['amount_log'] = np.log1p(X['amount']).astype('float32')

        # ratios per tx
        X['amount_to_orig_balance'] = (X['amount'] / (X['oldbalanceOrg'] + 1)).astype('float32')
        X['amount_to_dest_balance'] = (X['amount'] / (X['oldbalanceDest'] + 1)).astype('float32')

        # binary flags
        X['dest_account_empty_before'] = (X['oldbalanceDest'] == 0).astype('int8')

        return X

# legacy: this class will no longer be used, because the new dataset (PaySim) has different features and requires a 
#         different engineering approach. However, I am keeping it here for reference and potential reuse in other contexts.
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
