import pandas as pd
import numpy as np
import pytest
from src.features.engineering import PaySimFeatures


# ── Fixture: reusable minimal dataset ─────────────────────────────────────────
# A fixture is a function that prepares data for tests.
# @pytest.fixture allows pytest to automatically inject it into each test.

@pytest.fixture
def sample_df():
    """Minimal DataFrame with the same structure as the PaySim dataset."""
    return pd.DataFrame({
        "step":           [1, 25, 50],
        "type":           ["TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount":         [10000.0, 500.0, 999000.0],
        "nameOrig":       ["C001", "C002", "C001"],   # C001 appears twice → repeat
        "oldbalanceOrg":  [10000.0, 500.0, 999000.0],
        "newbalanceOrig": [0.0, 0.0, 0.0],
        "nameDest":       ["C100", "M200", "C300"],   # M200 is a merchant
        "oldbalanceDest": [0.0, 1000.0, 500.0],
        "newbalanceDest": [10000.0, 1500.0, 999500.0],
        "isFlaggedFraud": [0, 0, 0],
    })


# ── Test 1: transformer produces expected columns ─────────────────────────────
def test_transform_produces_expected_columns(sample_df):
    """Ensures that transform() generates the columns expected by the model."""
    fe = PaySimFeatures()
    fe.fit(sample_df)
    result = fe.transform(sample_df)

    expected_cols = {
        "is_transfer", "is_cash_out", "is_merchant_dest",
        "amount_log", "is_large_tx", "is_round_amount",
        "hour_of_day", "is_night",
        "orig_is_repeat", "dest_tx_count", "dest_unique_orig",
        "dest_was_empty", "amount_to_dest_ratio",
        "log_dest_balance", "log_orig_balance",
        # Columns that should not be removed from the original dataset:
        "oldbalanceOrg", "oldbalanceDest",
    }

    # Verifies that key engineered columns are present
    for col in ["is_transfer", "is_cash_out", "amount_log", "dest_was_empty"]:
        assert col in result.columns, f"Missing column: {col}"

    # Verifies that high-cardinality columns were removed
    for col in ["nameOrig", "nameDest", "type", "step", "amount"]:
        assert col not in result.columns, f"Column {col} should have been removed"


# ── Test 2: binary feature logic correctness ──────────────────────────────────
def test_binary_features_are_correct(sample_df):
    """Ensures that binary flags are computed according to business logic."""
    fe = PaySimFeatures()
    fe.fit(sample_df)
    result = fe.transform(sample_df)

    # Row 0: TRANSFER → is_transfer=1, is_cash_out=0
    assert result["is_transfer"].iloc[0] == 1
    assert result["is_cash_out"].iloc[0] == 0

    # Row 1: nameDest starts with "M" → is_merchant_dest=1
    assert result["is_merchant_dest"].iloc[1] == 1

    # Row 0: oldbalanceDest=0 → dest_was_empty=1
    assert result["dest_was_empty"].iloc[0] == 1

    # Row 1: oldbalanceDest=1000 → dest_was_empty=0
    assert result["dest_was_empty"].iloc[1] == 0

    # C001 appears twice → orig_is_repeat=1 for rows 0 and 2
    assert result["orig_is_repeat"].iloc[0] == 1
    assert result["orig_is_repeat"].iloc[1] == 0  # C002 appears only once


# ── Test 3: transform() without prior fit() should fail clearly ───────────────
def test_transform_without_fit_raises(sample_df):
    """
    Ensures that calling transform() before fit() raises an error,
    instead of producing silently incorrect results.
    """
    fe = PaySimFeatures()

    # threshold_large_ remains None until fit() is called.
    # An AttributeError or TypeError is expected when transform() is invoked.
    with pytest.raises((AttributeError, TypeError)):
        fe.transform(sample_df)