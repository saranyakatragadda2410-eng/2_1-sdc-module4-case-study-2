"""Feature selection using correlation and mutual information."""
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def correlation_ranking(df: pd.DataFrame, target: str) -> pd.Series:
    return df.corr(numeric_only=True)[target].abs().sort_values(ascending=False)


def mutual_info_ranking(X: pd.DataFrame, y) -> pd.Series:
    mi = mutual_info_classif(X, y, random_state=42)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)