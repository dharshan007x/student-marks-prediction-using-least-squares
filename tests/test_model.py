"""
test_model.py
=============
Unit tests for the LeastSquaresRegressor model.
Tests validate model fitting, coefficient shapes, design matrix,
and internal mathematical components.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.least_squares import LeastSquaresRegressor
from src.data_loader import (
    load_dataset, validate_dataset, get_features_and_target,
    split_data, FEATURE_COLUMNS, TARGET_COLUMN,
)
from src.utils import get_data_path


@pytest.fixture
def sample_data():
    """Small known dataset for testing."""
    X = np.array([
        [2.0, 60.0, 50.0, 45.0],
        [4.0, 70.0, 65.0, 55.0],
        [6.0, 80.0, 75.0, 70.0],
        [8.0, 85.0, 80.0, 75.0],
        [10.0, 90.0, 90.0, 85.0],
        [3.0, 65.0, 55.0, 50.0],
        [5.0, 75.0, 70.0, 60.0],
        [7.0, 82.0, 78.0, 72.0],
        [9.0, 88.0, 85.0, 80.0],
        [11.0, 95.0, 95.0, 90.0],
    ], dtype=np.float64)
    y = np.array([45.0, 55.0, 70.0, 78.0, 92.0,
                  48.0, 62.0, 74.0, 85.0, 96.0], dtype=np.float64)
    return X, y


@pytest.fixture
def fitted_model(sample_data):
    """Return a model fitted on sample data."""
    X, y = sample_data
    model = LeastSquaresRegressor(feature_names=FEATURE_COLUMNS)
    model.fit(X, y)
    return model


@pytest.fixture
def full_dataset():
    """Load the full project dataset."""
    data_path = get_data_path()
    df = load_dataset(data_path)
    return df


# ------------------------------------------------------------------
# Test 1: Model can fit valid student data
# ------------------------------------------------------------------
class TestModelFitting:
    def test_model_fits_valid_data(self, sample_data):
        X, y = sample_data
        model = LeastSquaresRegressor(feature_names=FEATURE_COLUMNS)
        model.fit(X, y)
        assert model.weights_ is not None

    # Test 2: Model has exactly four input features
    def test_four_input_features(self, fitted_model):
        assert fitted_model.n_features_ == 4

    # Test 3: Learned weight vector has correct shape
    def test_weight_vector_shape(self, fitted_model):
        # weights_ includes intercept: [w0, w1, w2, w3, w4] = 5 elements
        assert fitted_model.weights_.shape == (5,)

    # Test 4: Design matrix contains intercept column
    def test_design_matrix_intercept_column(self, sample_data, fitted_model):
        X, _ = sample_data
        X_design = fitted_model.get_design_matrix(X)
        # First column must be all ones
        np.testing.assert_array_equal(X_design[:, 0], np.ones(X.shape[0]))

    # Test 5: XtX has valid dimensions
    def test_xtx_dimensions(self, fitted_model):
        details = fitted_model.get_training_details()
        XtX = details["XtX"]
        # For D=4 features + 1 intercept -> (5, 5)
        assert XtX.shape == (5, 5)

    # Test 6: Xty has valid dimensions
    def test_xty_dimensions(self, fitted_model):
        details = fitted_model.get_training_details()
        Xty = details["Xty"]
        assert Xty.shape == (5,)


# ------------------------------------------------------------------
# Test 7-9: Predictions, Residuals, MSE
# ------------------------------------------------------------------
class TestPredictions:
    def test_predictions_correct_count(self, sample_data, fitted_model):
        X, _ = sample_data
        preds = fitted_model.predict(X)
        assert preds.shape == (X.shape[0],)

    # Test 8: Residuals have correct shape
    def test_residuals_shape(self, sample_data, fitted_model):
        X, y = sample_data
        res = fitted_model.residuals(X, y)
        assert res.shape == y.shape

    # Test 9: MSE is non-negative
    def test_mse_non_negative(self, sample_data, fitted_model):
        X, y = sample_data
        mse_val = fitted_model.mse(X, y)
        assert mse_val >= 0

    # Test 10: R2 is finite
    def test_r2_finite(self, sample_data, fitted_model):
        X, y = sample_data
        r2_val = fitted_model.r2_score(X, y)
        assert np.isfinite(r2_val)

    # Test 11: Predictions for valid student inputs are finite
    def test_single_prediction_finite(self, fitted_model):
        x_new = np.array([[7.5, 85.0, 72.0, 68.0]])
        pred = fitted_model.predict(x_new)
        assert np.all(np.isfinite(pred))

    # Test 12: Model does not crash on valid input
    def test_no_crash_on_valid_input(self, fitted_model):
        inputs = [
            np.array([[1.0, 40.0, 20.0, 20.0]]),
            np.array([[12.0, 100.0, 100.0, 100.0]]),
            np.array([[6.0, 70.0, 50.0, 50.0]]),
        ]
        for x in inputs:
            pred = fitted_model.predict(x)
            assert pred.shape == (1,)


# ------------------------------------------------------------------
# Test 13: Manual LS agrees with known solution for small dataset
# ------------------------------------------------------------------
class TestMathematicalCorrectness:
    def test_least_squares_agrees_with_numpy(self, sample_data):
        """
        Verify that the manually implemented Least Squares solution
        agrees with numpy.linalg.lstsq for a small known dataset.
        """
        X, y = sample_data

        # Manual implementation
        model = LeastSquaresRegressor(feature_names=FEATURE_COLUMNS)
        model.fit(X, y)

        # NumPy reference
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        w_ref, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

        np.testing.assert_allclose(model.weights_, w_ref, atol=1e-8)

    def test_model_equation_string(self, fitted_model):
        """Test that get_model_equation returns a string."""
        eq = fitted_model.get_model_equation()
        assert isinstance(eq, str)
        assert "Predicted Final Marks" in eq

    def test_intercept_matches_weights(self, fitted_model):
        """Intercept should be w[0]."""
        assert fitted_model.intercept_ == fitted_model.weights_[0]

    def test_coefficients_match_weights(self, fitted_model):
        """Coefficients should be w[1:]."""
        np.testing.assert_array_equal(
            fitted_model.coefficients_, fitted_model.weights_[1:]
        )
