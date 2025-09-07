from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    # Keep numeric columns only (simple baseline)
    num = df.select_dtypes(include="number").copy()
    if target_col not in num.columns:
        raise ValueError(f"Target column '{target_col}' not found in numeric columns.")
    num = num.dropna()
    X = num.drop(columns=[target_col])
    y = num[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)
