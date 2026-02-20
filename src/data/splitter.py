# src/data/splitter.py
from sklearn.model_selection import train_test_split
import pandas as pd
from src.config import RANDOM_SEED

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.
    Stratify=True by default because isFraud is heavily imbalanced.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y if stratify else None,
    )