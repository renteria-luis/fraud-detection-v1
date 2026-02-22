# src/data/splitter.py
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from src.config import RANDOM_SEED, PROCESSED_DIR

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

def save_splits(X_train, X_test, y_train, y_test, path=PROCESSED_DIR):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(path / "X_train.parquet")
    X_test.to_parquet(path / "X_test.parquet")
    y_train.to_frame().to_parquet(path / "y_train.parquet")
    y_test.to_frame().to_parquet(path / "y_test.parquet")

def load_splits(path=PROCESSED_DIR):
    path = Path(path)

    return (
        pd.read_parquet(path / "X_train.parquet"),
        pd.read_parquet(path / "X_test.parquet"),
        pd.read_parquet(path / "y_train.parquet").squeeze(),
        pd.read_parquet(path / "y_test.parquet").squeeze(),
    )
