import numpy as np
import pandas as pd
import pytest

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


@pytest.fixture
def sample_data():
    """Provide a small labeled dataset for testing."""
    return pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55],
        "workclass": [
            "Private", "Private", "Government", "Government",
            "Private", "Private", "Government", "Government",
        ],
        "salary": [
            "<=50K", "<=50K", "<=50K", "<=50K",
            ">50K", ">50K", ">50K", ">50K",
        ],
    })


def test_train_and_inference(sample_data):
    """Check that training learns the simple salary pattern."""
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=["workclass"],
        label="salary",
    )
    model = train_model(X, y)
    predictions = inference(model, X)

    assert predictions.shape == y.shape
    np.testing.assert_array_equal(predictions, y)


def test_compute_model_metrics():
    """Check metrics against manually calculated values."""
    y = np.array([1, 1, 0, 0])
    predictions = np.array([1, 0, 1, 0])

    precision, recall, f1 = compute_model_metrics(y, predictions)

    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_save_and_load(sample_data, tmp_path):
    """Check that saved model and encoders preserve predictions."""
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=["workclass"],
        label="salary",
    )
    model = train_model(X, y)

    objects = {
        "model": model,
        "encoder": encoder,
        "lb": lb,
    }
    loaded = {}

    for name, obj in objects.items():
        path = tmp_path / f"{name}.pkl"
        save_model(obj, path)
        loaded[name] = load_model(path)

    X_loaded, y_loaded, _, _ = process_data(
        sample_data,
        categorical_features=["workclass"],
        label="salary",
        training=False,
        encoder=loaded["encoder"],
        lb=loaded["lb"],
    )

    np.testing.assert_array_equal(X_loaded, X)
    np.testing.assert_array_equal(y_loaded, y)
    np.testing.assert_array_equal(
        inference(loaded["model"], X_loaded),
        inference(model, X),
    )


def test_process_unseen_category(sample_data):
    """Check unlabeled inference with an unseen category."""
    X_train, _, encoder, lb = process_data(
        sample_data,
        categorical_features=["workclass"],
        label="salary",
    )
    new_data = pd.DataFrame({
        "age": [32],
        "workclass": ["Self-employed"],
    })

    X, y, _, _ = process_data(
        new_data,
        categorical_features=["workclass"],
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert X.shape == (1, X_train.shape[1])
    assert y.size == 0
    assert X[0, 0] == 32
    np.testing.assert_array_equal(X[0, 1:], [0, 0])


def test_categorical_slice(sample_data):
    """Check that slice metrics use only the selected rows."""
    _, _, encoder, lb = process_data(
        sample_data,
        categorical_features=["workclass"],
        label="salary",
    )

    class PredictPositive:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

    precision, recall, f1 = performance_on_categorical_slice(
        sample_data,
        "workclass",
        "Private",
        ["workclass"],
        "salary",
        encoder,
        lb,
        PredictPositive(),
    )

    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(2 / 3)
