import pandas as pd

def load_transaction_data(path: str) -> pd.DataFrame:
    """
    Load PaySim dataset with optimized data types for memory efficiency.
    Excludes (for not being useful for modeling):
        - isFlaggedFraud: Only 0.001% of transactions are flagged.
        - nameOrig and nameDest: Categorical IDs with high cardinality.
    """
    dtypes = {
        'step': 'int16',
        'type': 'category',
        'amount': 'float32',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'uint8',
    }
    
    return pd.read_csv(path, dtype=dtypes, usecols=list(dtypes.keys()))
