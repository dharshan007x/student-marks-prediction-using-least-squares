"""
test_math.py
============
Tests for mathematical correctness of the Least Squares implementation.
Verifies normal equation components, design matrix construction,
and numerical accuracy against known analytical solutions.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.least_squares import LeastSquaresRegressor


class TestDesignMatrix:
    def test_intercept_column_ones(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0]])
        model = LeastSquaresRegressor()
        X_design = model.get_design_matrix(X)
        np.testing.assert_array_equal(X_design[:, 0], [1.0, 1.0])

    def test_design_matrix_shape(self):
        X = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]], dtype=float)
        model = LeastSquaresRegressor()
        X_design = model.get_design_matrix(X)
        assert X_design.shape == (3, 5)

    def test_design_matrix_preserves_features(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        model = LeastSquaresRegressor()
        X_design = model.get_design_matrix(X)
        np.testing.assert_array_equal(X_design[0, 1:], [1.0, 2.0, 3.0, 4.0])

    def test_1d_input_reshaped(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        model = LeastSquaresRegressor()
        X_design = model.get_design_matrix(x)
        assert X_design.shape == (1, 5)


class TestNormalEquations:
    def test_xtx_symmetric(self):
        """X^T X must be symmetric."""
        X = np.array([[2, 60, 50, 45],
                       [4, 70, 65, 55],
                       [6, 80, 75, 70]], dtype=float)
        y = np.array([50, 65, 80], dtype=float)
        model = LeastSquaresRegressor()
        comps = model.get_normal_equation_components(X, y)
        XtX = comps["XtX"]
        np.testing.assert_allclose(XtX, XtX.T, atol=1e-10)

    def test_xtx_positive_semidefinite(self):
        """X^T X should be positive semi-definite (all eigenvalues >= 0)."""
        X = np.array([[2, 60, 50, 45],
                       [4, 70, 65, 55],
                       [6, 80, 75, 70],
                       [8, 85, 80, 75],
                       [10, 90, 90, 85]], dtype=float)
        y = np.array([50, 65, 80, 85, 95], dtype=float)
        model = LeastSquaresRegressor()
        comps = model.get_normal_equation_components(X, y)
        eigenvalues = np.linalg.eigvalsh(comps["XtX"])
        assert np.all(eigenvalues >= -1e-10)

    def test_normal_equation_solution(self):
        """Verify that w solves X^T X w = X^T y."""
        X = np.array([[2, 60, 50, 45],
                       [4, 70, 65, 55],
                       [6, 80, 75, 70],
                       [8, 85, 80, 75],
                       [10, 90, 90, 85]], dtype=float)
        y = np.array([50, 65, 80, 85, 95], dtype=float)
        model = LeastSquaresRegressor()
        model.fit(X, y)

        X_design = model.get_design_matrix(X)
        lhs = X_design.T @ X_design @ model.weights_
        rhs = X_design.T @ y
        np.testing.assert_allclose(lhs, rhs, atol=1e-6)


class TestSimpleLinearCase:
    """Test with a perfectly linear dataset (no noise)."""

    def test_perfect_fit_r2(self):
        # y = 10 + 2*x1 + 0.5*x2 + 0.3*x3 + 0.1*x4
        X = np.array([
            [1, 10, 20, 30],
            [2, 20, 30, 40],
            [3, 30, 40, 50],
            [4, 40, 50, 60],
            [5, 50, 60, 70],
        ], dtype=float)
        y = 10 + 2 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.1 * X[:, 3]

        model = LeastSquaresRegressor()
        model.fit(X, y)

        # R2 should be 1.0 for perfect linear data
        r2 = model.r2_score(X, y)
        np.testing.assert_almost_equal(r2, 1.0, decimal=10)

    def test_perfect_fit_mse(self):
        X = np.array([
            [1, 10, 20, 30],
            [2, 20, 30, 40],
            [3, 30, 40, 50],
            [4, 40, 50, 60],
            [5, 50, 60, 70],
        ], dtype=float)
        y = 10 + 2 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.1 * X[:, 3]

        model = LeastSquaresRegressor()
        model.fit(X, y)

        mse = model.mse(X, y)
        np.testing.assert_almost_equal(mse, 0.0, decimal=10)

    def test_perfect_fit_recovers_coefficients(self):
        # Non-collinear features so XtX is invertible
        rng = np.random.RandomState(99)
        X = rng.uniform(1, 10, (20, 4))
        true_w = np.array([10.0, 2.0, 0.5, 0.3, 0.1])
        y = true_w[0] + X @ true_w[1:]

        model = LeastSquaresRegressor()
        model.fit(X, y)

        np.testing.assert_allclose(model.weights_, true_w, atol=1e-8)


class TestSumSquaredError:
    def test_sse_matches_manual(self):
        X = np.array([[2, 60, 50, 45],
                       [4, 70, 65, 55],
                       [6, 80, 75, 70]], dtype=float)
        y = np.array([50, 65, 80], dtype=float)

        model = LeastSquaresRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        manual_sse = 0.5 * np.sum((y - preds) ** 2)
        model_sse = model.sum_squared_error(X, y)
        np.testing.assert_almost_equal(model_sse, manual_sse)
