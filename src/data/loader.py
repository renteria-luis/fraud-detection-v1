# src/data/loader.py
import pandas as pd
from src.config import PAYSIM_PATH, FRAUD_TYPES, DROP_COLS, TARGET

_DTYPES = {
    'step':           'int16',
    'type':           'category',
    'amount':         'float32',
    'nameOrig':       'object',
    'oldbalanceOrg':  'float32',
    'newbalanceOrig': 'float32',
    'nameDest':       'object',
    'oldbalanceDest': 'float32',
    'newbalanceDest': 'float32',
    'isFraud':        'uint8',
    'isFlaggedFraud': 'uint8',
}

def load_paysim(path=PAYSIM_PATH) -> pd.DataFrame:
    """
    Load PaySim raw CSV with optimized dtypes.
    Returns full dataframe before any filtering.
    """
    return pd.read_csv(path, dtype=_DTYPES)


def filter_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply business rules:
      - Keep only TRANSFER and CASH_OUT (only types with fraud)
      - Drop leakage and high-cardinality columns
      - Split features from target
    Returns (X, y).
    """
    max_legit_step = df[df[TARGET] == 0]['step'].max()
    df = df[df['step'] <= max_legit_step].copy()
    
    df = df[df['type'].isin(FRAUD_TYPES)].copy()
    df = df.drop(columns=DROP_COLS)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
