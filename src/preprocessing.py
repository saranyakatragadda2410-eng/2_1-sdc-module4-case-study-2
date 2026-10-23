"""Data preprocessing helpers."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fill missing ages with median
    if df["Age"].isna().any():
        df["Age"] = df["Age"].fillna(df["Age"].median())
    # Accept either column name, then use Gender consistently.
    if "Gender" not in df.columns and "Sex" in df.columns:
        df = df.rename(columns={"Sex": "Gender"})
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    # Encode Embarked
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])
    return df


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def encode_labels(y):
    return LabelEncoder().fit_transform(y)


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.3, random_state: int = 42):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)