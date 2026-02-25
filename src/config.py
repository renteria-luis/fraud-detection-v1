# src/config.py
from pathlib import Path

ROOT = Path(__file__).parent.parent
PAYSIM_PATH   = ROOT / 'data' / 'raw' / 'PS_20174392719_1491204439457_log.csv'
PROCESSED_DIR = ROOT / 'data' / 'processed'
MODELS_DIR    = ROOT / 'models'

FRAUD_TYPES = ['TRANSFER', 'CASH_OUT']
TARGET      = 'isFraud'

DROP_COLS = [
    'newbalanceOrig',
    'newbalanceDest',
    # 'oldbalanceOrg', testing intentional leakage removal
    # 'oldbalanceDest', testing intentional leakage removal
    'isFlaggedFraud',
]

NUMERIC_FEATURES = [
    'amount_log',
    'hour_of_day',
    'dest_tx_count',
    'dest_unique_orig',
    'amount_to_dest_ratio',
    'log_dest_balance',
    'log_orig_balance',
]

BINARY_FEATURES = [
    'is_transfer',
    'is_cash_out',
    'is_merchant_dest',
    'is_large_tx',
    'is_round_amount',
    'is_night',
    'orig_is_repeat',
    'dest_was_empty',  # testing intentional leakage removal
]

CYCLICAL_FEATURES = ['hour_sin', 'hour_cos']
RANDOM_SEED = 42