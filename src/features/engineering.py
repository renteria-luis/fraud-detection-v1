# src/features/engineering.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

class PaySimFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, cyclical_encoding: bool = False, large_tx_quantile: float = 0.95):
        self.cyclical_encoding  = cyclical_encoding
        self.large_tx_quantile  = large_tx_quantile
        self.threshold_large_   = None
        self.dest_tx_count_     = None  # dict
        self.dest_unique_orig_  = None  # dict
        self.orig_is_repeat_    = None  # dict

    def fit(self, X: pd.DataFrame, y=None):
        # amount threshold
        self.threshold_large_ = X['amount'].quantile(self.large_tx_quantile)

        # nameOrig
        orig_counts = X['nameOrig'].value_counts()
        self.orig_is_repeat_ = (orig_counts > 1).to_dict()

        # nameDest aggs
        dest_groups = X.groupby('nameDest')
        self.dest_tx_count_    = dest_groups['amount'].count().to_dict()
        self.dest_unique_orig_ = dest_groups['nameOrig'].nunique().to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # TYPE
        X['is_transfer']  = (X['type'] == 'TRANSFER').astype('int8')
        X['is_cash_out']  = (X['type'] == 'CASH_OUT').astype('int8')

        # DEST
        X['is_merchant_dest'] = X['nameDest'].str.startswith('M').astype('int8')

        # AMOUNT
        X['amount_log'] = np.log1p(X['amount']).astype('float32')
        X['is_large_tx'] = (X['amount'] > self.threshold_large_).astype('int8')
        X['is_round_amount'] = ((X['amount'] % 1000) == 0).astype('int8')

        # TIME
        hour = (X['step'] % 24).astype('int8')
        X['hour_of_day'] = hour
        X['is_night'] = hour.between(0, 6).astype('int8')

        if self.cyclical_encoding:
            X['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype('float32')
            X['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype('float32')

        # ORIG, rep in origin
        X['orig_is_repeat'] = (
            X['nameOrig']
            .map(self.orig_is_repeat_)
            .infer_objects(copy=False)
            .fillna(0)
            .astype('int8')
        )

        # DEST aggs
        X['dest_tx_count']    = X['nameDest'].map(self.dest_tx_count_).fillna(1).astype('float32')
        X['dest_unique_orig'] = X['nameDest'].map(self.dest_unique_orig_).fillna(1).astype('float32')

        # Testing
        X['dest_was_empty'] = (X['oldbalanceDest'] == 0).astype('int8')
        X['amount_to_dest_ratio'] = (X['amount'] / (X['oldbalanceDest'] + 1)).astype('float32')
        X['log_dest_balance']     = np.log1p(X['oldbalanceDest']).astype('float32')
        X['log_orig_balance'] = np.log1p(X['oldbalanceOrg']).astype('float32')

        # DROP
        X = X.drop(columns=['step', 'type', 'nameOrig', 'nameDest', 'amount', 'oldbalanceDest', 'oldbalanceOrg'])

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
