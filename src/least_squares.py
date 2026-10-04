"""
least_squares.py
================
Manual implementation of the Least Squares Regression model using
the Normal Equations.

This is the PRIMARY algorithm for the project:
    "Student Marks Prediction Using Least Squares"

The implementation is fully transparent — every mathematical step
(design matrix, XᵀX, Xᵀy, coefficient estimation) is exposed so
that it can be inspected during an academic viva.

Mathematical Background
-----------------------
Given N training samples with D features, we form:

    X  : (N, D+1) design matrix (column of 1s prepended for intercept)
    t  : (N,)     target vector  (Final Examination Marks)

The linear model is:

    y(x, w) = w₀ + w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄  =  Xw

The Sum-of-Squares Error:

    E(w) = ½ Σₙ [y(xₙ, w) − tₙ]²

Setting ∂E/∂w = 0 yields the Normal Equations:

    XᵀX w = Xᵀt

Solved via:

    w = (XᵀX)⁻¹ Xᵀt

We use numpy.linalg.solve for numerical stability instead of
explicitly computing the inverse.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class LeastSquaresRegressor:
    """
    Multiple Linear Regression via Ordinary Least Squares.

    Attributes
    ----------
    weights_ : np.ndarray or None
        Full weight vector [w₀, w₁, …, w_D] after fitting.
    intercept_ : float or None
        The bias / intercept term w₀.
    coefficients_ : np.ndarray or None
        Regression coefficients [w₁, …, w_D] (excludes intercept).
    feature_names_ : list[str] or None
        Names of the input features.
    n_features_ : int or None
        Number of input features (D).
    training_error_ : float or None
        Sum-of-Squares Error on the training set.
    training_mse_ : float or None
        Mean Squared Error on the training set.
    training_r2_ : float or None
        R² score on the training set.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = feature_names
        self.n_features_: Optional[int] = None
        self.training_error_: Optional[float] = None
        self.training_mse_: Optional[float] = None
        self.training_r2_: Optional[float] = None

        # Internal storage for viva inspection
        self._X_design: Optional[np.ndarray] = None  # Design matrix used in training
        self._XtX: Optional[np.ndarray] = None
        self._Xty: Optional[np.ndarray] = None
        self._train_predictions: Optional[np.ndarray] = None
        self._train_residuals: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LeastSquaresRegressor":
        """
        Fit the Least Squares model using the Normal Equations.

        Steps performed:
            1. Validate input dimensions.
            2. Add the bias/intercept column (column of 1s).
            3. Construct the design matrix  X_design.
            4. Calculate  XᵀX.
            5. Calculate  Xᵀy.
            6. Solve the Normal Equations:  (XᵀX) w = Xᵀy.
            7. Store the learned coefficients.
            8. Generate training predictions.
            9. Calculate training residuals.
           10. Calculate the Sum-of-Squares Error.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Feature matrix (WITHOUT intercept column).
        y : np.ndarray, shape (N,)
            Target vector.

        Returns
        -------
        self
        """
        # Step 1 — Validate
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}-D")
        N, D = X.shape
        if y.shape[0] != N:
            raise ValueError(
                f"X has {N} samples but y has {y.shape[0]} values"
            )

        self.n_features_ = D
        if self.feature_names_ is None:
            self.feature_names_ = [f"x{i+1}" for i in range(D)]

        # Step 2 & 3 — Design matrix with intercept
        X_design = self.get_design_matrix(X)
        self._X_design = X_design

        # Step 4 — XᵀX
        XtX = X_design.T @ X_design
        self._XtX = XtX

        # Step 5 — Xᵀy
        Xty = X_design.T @ y
        self._Xty = Xty

        # Step 6 — Solve the normal equations using numpy.linalg.solve
        #          This avoids explicitly computing the matrix inverse
        #          and is numerically more stable.
        #          Falls back to lstsq if XtX is singular (rank-deficient).
        try:
            w = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

        # Step 7 — Store coefficients
        self.weights_ = w
        self.intercept_ = w[0]
        self.coefficients_ = w[1:]

        # Step 8 — Training predictions
        self._train_predictions = X_design @ w

        # Step 9 — Training residuals
        self._train_residuals = y - self._train_predictions

        # Step 10 — Sum-of-Squares Error (with the ½ factor)
        self.training_error_ = 0.5 * np.sum(self._train_residuals ** 2)

        # Also store training MSE and R²
        self.training_mse_ = float(np.mean(self._train_residuals ** 2))
        ss_res = np.sum(self._train_residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.training_r2_ = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new data.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Feature matrix (WITHOUT intercept column).

        Returns
        -------
        np.ndarray, shape (N,)
            Predicted Final Examination Marks.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_design = self.get_design_matrix(X)
        return X_design @ self.weights_

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate residuals:  r = y − ŷ  (actual − predicted).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Actual target values.

        Returns
        -------
        np.ndarray
        """
        y_pred = self.predict(X)
        return y - y_pred

    def sum_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the Sum-of-Squares Error (with factor ½):

            E(w) = ½ Σₙ (yₙ − tₙ)²

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Actual target values.

        Returns
        -------
        float
        """
        res = self.residuals(X, y)
        return float(0.5 * np.sum(res ** 2))

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Mean Squared Error:

            MSE = (1/N) Σ (tₙ − yₙ)²

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Actual target values.

        Returns
        -------
        float
        """
        res = self.residuals(X, y)
        return float(np.mean(res ** 2))

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        R² (Coefficient of Determination):

            R² = 1 − [Σ(tₙ − yₙ)² / Σ(tₙ − t̄)²]

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Actual target values.

        Returns
        -------
        float
        """
        y = np.asarray(y, dtype=np.float64)
        res = self.residuals(X, y)
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def get_model_equation(self) -> str:
        """
        Return a human-readable string of the learned regression equation.

        Returns
        -------
        str
            e.g. "Predicted Final Marks = 12.34 + 4.21(Study_Hours) + ..."
        """
        self._check_fitted()
        parts = [f"{self.intercept_:+.4f}"]
        for name, coef in zip(self.feature_names_, self.coefficients_):
            parts.append(f"{coef:+.4f} × {name}")
        equation = "Predicted Final Marks = " + " ".join(parts)
        return equation

    def get_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construct the design matrix by prepending a column of 1s.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        np.ndarray, shape (N, D+1)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack([ones, X])

    def get_normal_equation_components(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate and return all Normal Equation components for
        academic inspection.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Feature matrix (without intercept).
        y : np.ndarray, shape (N,)
            Target vector.

        Returns
        -------
        dict with keys:
            'X_design', 'Xt', 'XtX', 'Xty', 'weights'
        """
        X_design = self.get_design_matrix(X)
        Xt = X_design.T
        XtX = Xt @ X_design
        Xty = Xt @ y
        w = np.linalg.solve(XtX, Xty)
        return {
            "X_design": X_design,
            "Xt": Xt,
            "XtX": XtX,
            "Xty": Xty,
            "weights": w,
        }

    def get_training_details(self) -> Dict:
        """
        Return a dictionary of internal training details for viva
        demonstration.

        Returns
        -------
        dict
        """
        self._check_fitted()
        return {
            "n_features": self.n_features_,
            "feature_names": self.feature_names_,
            "weights": self.weights_,
            "intercept": self.intercept_,
            "coefficients": self.coefficients_,
            "training_sse": self.training_error_,
            "training_mse": self.training_mse_,
            "training_r2": self.training_r2_,
            "design_matrix_shape": self._X_design.shape if self._X_design is not None else None,
            "XtX": self._XtX,
            "Xty": self._Xty,
            "training_predictions": self._train_predictions,
            "training_residuals": self._train_residuals,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.weights_ is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit(X, y) first."
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.weights_ is not None else "not fitted"
        n = self.n_features_ if self.n_features_ is not None else "?"
        return f"LeastSquaresRegressor(n_features={n}, status={status})"
