"""
test_predictions.py
===================
Tests for prediction functionality and evaluation metrics.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.least_squares import LeastSquaresRegressor
from src.evaluation import (
    mean_squared_error, r2_score, mean_absolute_error,
    root_mean_squared_error, residual_statistics,
)
from src.data_loader import FEATURE_COLUMNS


@pytest.fixture
def trained_model():
    """A model trained on synthetic data."""
    rng = np.random.RandomState(123)
    X = rng.uniform(1, 12, (50, 4))
    y = 5.0 + 3.0 * X[:, 0] + 0.1 * X[:, 1] + 0.3 * X[:, 2] + 0.4 * X[:, 3]
    y += rng.normal(0, 2, 50)
    model = LeastSquaresRegressor(feature_names=FEATURE_COLUMNS)
    model.fit(X, y)
    return model, X, y


class TestPredictionOutput:
    def test_predict_returns_ndarray(self, trained_model):
        model, X, _ = trained_model
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_single_returns_scalar(self, trained_model):
        model, _, _ = trained_model
        x_single = np.array([[5.0, 70.0, 60.0, 55.0]])
        pred = model.predict(x_single)
        assert pred.shape == (1,)

    def test_predict_multiple(self, trained_model):
        model, _, _ = trained_model
        X_new = np.array([
            [3.0, 60.0, 50.0, 40.0],
            [6.0, 75.0, 70.0, 65.0],
            [9.0, 90.0, 85.0, 80.0],
        ])
        preds = model.predict(X_new)
        assert preds.shape == (3,)
        assert np.all(np.isfinite(preds))


class TestEvaluationMetrics:
    def test_mse_zero_for_identical(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == 0.0

    def test_mse_positive_for_different(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert mean_squared_error(y_true, y_pred) > 0

    def test_r2_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_score(y, y) == 1.0

    def test_r2_reasonable_range(self, trained_model):
        model, X, y = trained_model
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        assert -1.0 <= r2 <= 1.0

    def test_mae_non_negative(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        assert mean_absolute_error(y_true, y_pred) >= 0

    def test_rmse_equals_sqrt_mse(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        np.testing.assert_almost_equal(rmse, np.sqrt(mse))

    def test_residual_statistics_keys(self, trained_model):
        model, X, y = trained_model
        preds = model.predict(X)
        stats = residual_statistics(y, preds)
        expected_keys = {"mean", "std", "min", "max", "median", "q25", "q75", "count"}
        assert expected_keys == set(stats.keys())


class TestModelNotFitted:
    def test_predict_before_fit_raises(self):
        model = LeastSquaresRegressor()
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1, 2, 3, 4]]))

    def test_residuals_before_fit_raises(self):
        model = LeastSquaresRegressor()
        with pytest.raises(RuntimeError):
            model.residuals(np.array([[1, 2, 3, 4]]), np.array([5]))
