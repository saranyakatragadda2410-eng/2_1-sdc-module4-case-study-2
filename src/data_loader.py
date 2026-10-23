"""Load Iris and Titanic datasets."""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_iris(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "iris_50.csv")
    df = pd.read_csv(path)
    print(f"[Iris] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(df["species"].value_counts().to_dict())
    return df


def load_titanic(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "titanic_50.csv")
    df = pd.read_csv(path)
    print(f"[Titanic] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Survival rate: {df['Survived'].mean():.2f}")
    return df


if __name__ == "__main__":
    load_iris()
    load_titanic()