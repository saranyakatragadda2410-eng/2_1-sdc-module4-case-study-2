"""Iris classification pipeline."""
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.data_loader import load_iris
from src.preprocessing import split_data, scale_features, encode_labels
from src.model_evaluation import evaluate_model
from src.feature_selection import mutual_info_ranking


def run_iris_pipeline() -> list:
    df = load_iris()
    X_train, X_test, y_train, y_test = split_data(df, target="species", test_size=0.3)
    y_train_enc = encode_labels(y_train)
    y_test_enc = encode_labels(y_test)
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)

    print("\nFeature importance (mutual information):")
    print(mutual_info_ranking(X_train, y_train_enc))

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=3),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
    }

    results = []
    for name, model in models.items():
        cv = cross_val_score(model, X_train_s, y_train_enc, cv=3).mean()
        model.fit(X_train_s, y_train_enc)
        metrics = evaluate_model(name, model, X_test_s, y_test_enc)
        metrics["cv_mean"] = cv
        results.append(metrics)
    return results


if __name__ == "__main__":
    run_iris_pipeline()