# src/config.py
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Data
PAYSIM_PATH = ROOT / 'data' / 'raw' / 'PS_20174392719_1491204439457_log.csv'
FRAUD_TYPES = ['TRANSFER', 'CASH_OUT']   # unique values in 'type' column with isFraud == 1

# Columns
TARGET = 'isFraud'
DROP_COLS = [
    'newbalanceOrig',   # leakage: not available at transaction time
    'newbalanceDest',   # leakage: not available at transaction time
    'isFlaggedFraud',   # target leak
    'nameOrig',         # high cardinality, not useful for modeling
    'nameDest',         # high cardinality, not useful for modeling
]

# Expected features
NUMERIC_FEATURES = [
    'amount',
    'amount_log',
    'oldbalanceOrg',
    'oldbalanceDest',
    'hour_of_day',
    'day',
    'amount_to_orig_balance',
    'amount_to_dest_balance',
]
BINARY_FEATURES = ['dest_account_empty_before']
CATEGORICAL_FEATURES = ['type']

# if cyclical_encoding == True:
CYCLICAL_FEATURES = ['hour_sin', 'hour_cos']

# Model
RANDOM_SEED = 42