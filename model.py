import json
from datetime import datetime, timezone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_data():
    """Load the Iris dataset and return feature matrix, labels, and metadata."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = [str(name) for name in iris.target_names]
    return X, y, feature_names, target_names


def train_model(X_train, y_train, random_state=42, n_estimators=100):
    """Train a RandomForestClassifier with a fixed random seed."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=4,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    """Evaluate model and return accuracy, predictions, and detailed metrics."""
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return accuracy, y_pred, report


def save_metrics(metrics, filepath="metrics.json"):
    """Persist evaluation metrics to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Saved metrics to {filepath}")


def run_pipeline():
    """Execute complete ML training, evaluation, and logging pipeline."""
    model_name = "Random Forest Classifier"
    dataset_name = "Iris Dataset"

    print("=" * 60)
    print(" MACHINE LEARNING MODEL PIPELINE - CLA-3 ASSIGNMENT")
    print("=" * 60)
    print(f"Model Name   : {model_name}")
    print(f"Dataset Name : {dataset_name}")

    # Load data
    X, y, feature_names, target_names = load_data()
    print(f"Total Samples: {len(X)} | Features: {len(feature_names)} | Classes: {len(target_names)}")

    # Stratified train/test split with fixed random seed
    random_seed = 42
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    print(f"Split        : Train={len(X_train)} samples, Test={len(X_test)} samples (Stratified)")

    # Train model
    clf = train_model(X_train, y_train, random_state=random_seed)

    # Evaluate
    accuracy, y_pred, report = evaluate_model(clf, X_test, y_test)

    # Print predictions and metrics
    print("\n[PREDICTIONS on Test Set]")
    print(f"Ground Truth : {y_test.tolist()}")
    print(f"Predictions  : {y_pred.tolist()}")
    print(f"\n[EVALUATION METRICS]")
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({accuracy:.4f})")

    # Prepare metrics dictionary
    metrics = {
        "project_name": "CI/CD Pipeline for Machine Learning Model",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "algorithm": "RandomForestClassifier",
        "num_total_samples": len(X),
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "test_size": test_size,
        "random_seed": random_seed,
        "accuracy": round(accuracy, 4),
        "accuracy_percentage": f"{accuracy * 100:.2f}%",
        "feature_names": feature_names,
        "target_classes": target_names,
        "test_ground_truth": [int(val) for val in y_test],
        "test_predictions": [int(val) for val in y_pred],
        "macro_avg_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_avg_f1": round(report["weighted avg"]["f1-score"], 4),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS"
    }

    save_metrics(metrics, "metrics.json")
    print("=" * 60)
    return metrics, clf


if __name__ == "__main__":
    run_pipeline()
