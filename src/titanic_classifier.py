"""Titanic survival classification pipeline."""
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.data_loader import load_titanic
from src.preprocessing import clean_titanic, split_data, scale_features
from src.model_evaluation import evaluate_model
from src.feature_selection import correlation_ranking, mutual_info_ranking


def run_titanic_pipeline() -> list:
    df = load_titanic()
    df = clean_titanic(df)
    df = df.drop(columns=["PassengerId"])

    X_train, X_test, y_train, y_test = split_data(df, target="Survived", test_size=0.3)

    print("\nFeature correlation with Survived:")
    print(correlation_ranking(df, "Survived"))
    print("\nMutual information:")
    print(mutual_info_ranking(X_train, y_train))

    X_train_s, X_test_s, _ = scale_features(X_train, X_test)

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=500),
    }

    results = []
    for name, model in models.items():
        cv = cross_val_score(model, X_train_s, y_train, cv=3).mean()
        model.fit(X_train_s, y_train)
        metrics = evaluate_model(name, model, X_test_s, y_test)
        metrics["cv_mean"] = cv
        results.append(metrics)
    return results


if __name__ == "__main__":
    run_titanic_pipeline()