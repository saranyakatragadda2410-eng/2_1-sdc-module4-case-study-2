"""Model evaluation utilities."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
    }
    print(f"\n--- {name} ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    return metrics