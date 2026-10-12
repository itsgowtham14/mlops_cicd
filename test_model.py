import os
import json
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model import load_data, train_model, evaluate_model, save_metrics, run_pipeline


def test_dataset_loading():
    """Verify Iris dataset loads properly with correct shape and classes."""
    X, y, feature_names, target_names = load_data()
    assert X is not None, "Feature matrix X should not be None"
    assert y is not None, "Target labels y should not be None"
    assert X.shape == (150, 4), f"Expected shape (150, 4), got {X.shape}"
    assert len(y) == 150, f"Expected 150 labels, got {len(y)}"
    assert len(np.unique(y)) == 3, "Expected 3 distinct classes in Iris dataset"
    assert len(feature_names) == 4, "Expected 4 feature names"
    assert len(target_names) == 3, "Expected 3 target names"


def test_model_training():
    """Verify model training produces a fitted RandomForestClassifier."""
    X, y, _, _ = load_data()
    clf = train_model(X[:100], y[:100], random_state=42)
    assert isinstance(clf, RandomForestClassifier), "Model should be RandomForestClassifier"
    assert hasattr(clf, "estimators_"), "Model should be fitted"
    assert len(clf.estimators_) == 100, "Model should have 100 trees"


def test_prediction_output():
    """Verify prediction output shape, types, and valid class values."""
    X, y, _, _ = load_data()
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]
    clf = train_model(X_train, y_train, random_state=42)
    accuracy, y_pred, _ = evaluate_model(clf, X_test, y_test)

    assert len(y_pred) == len(X_test), "Predictions count must match test size"
    for pred in y_pred:
        assert pred in [0, 1, 2], f"Invalid predicted class label: {pred}"


def test_model_accuracy_threshold():
    """Verify model achieves acceptable classification accuracy (>= 90%)."""
    metrics, _ = run_pipeline()
    assert "accuracy" in metrics, "Metrics must include accuracy"
    assert metrics["accuracy"] >= 0.90, f"Accuracy too low: {metrics['accuracy']}"
    assert metrics["model_name"] == "Random Forest Classifier"
    assert metrics["dataset_name"] == "Iris Dataset"


def test_metrics_json_persisted():
    """Verify metrics.json is generated and contains required metadata."""
    if os.path.exists("metrics.json"):
        os.remove("metrics.json")

    run_pipeline()
    assert os.path.exists("metrics.json"), "metrics.json must be created"

    with open("metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "project_name",
        "model_name",
        "dataset_name",
        "algorithm",
        "num_total_samples",
        "num_train_samples",
        "num_test_samples",
        "accuracy",
        "accuracy_percentage",
        "random_seed",
        "status"
    ]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in metrics.json"

    assert 0.0 <= data["accuracy"] <= 1.0, f"Invalid accuracy value: {data['accuracy']}"
