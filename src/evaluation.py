"""
evaluation.py
=============
Evaluation utilities for the Student Marks Prediction project.

Provides manual implementations of MSE and R² along with
helper functions for residual analysis and model comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE) manually.

        MSE = (1/N) Σᵢ (tᵢ − yᵢ)²

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (Coefficient of Determination) manually.

        R² = 1 − [Σ(tᵢ − yᵢ)² / Σ(tᵢ − t̄)²]

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE) — optional additional metric.

        MAE = (1/N) Σᵢ |tᵢ − yᵢ|

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE) — optional additional metric.

        RMSE = √MSE

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    float
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def residual_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute residual statistics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    dict
        mean, std, min, max, median, q25, q75
    """
    residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "q25": float(np.percentile(residuals, 25)),
        "q75": float(np.percentile(residuals, 75)),
        "count": len(residuals),
    }


def residual_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_largest: int = 10,
) -> pd.DataFrame:
    """
    Build a DataFrame of actual, predicted, and residual values,
    sorted by absolute residual descending.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    n_largest : int
        Number of rows to return (largest absolute residuals).

    Returns
    -------
    pd.DataFrame
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    df = pd.DataFrame({
        "Actual": np.round(y_true, 2),
        "Predicted": np.round(y_pred, 2),
        "Residual": np.round(residuals, 2),
        "Abs_Residual": np.round(np.abs(residuals), 2),
    })
    df = df.sort_values("Abs_Residual", ascending=False).reset_index(drop=True)
    if n_largest > 0:
        return df.head(n_largest)
    return df


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Comprehensive evaluation returning all metrics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    dict
    """
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "residual_stats": residual_statistics(y_true, y_pred),
    }


def compare_models(
    results: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Compare multiple model evaluations in a table.

    Parameters
    ----------
    results : dict
        {model_name: {metric_name: value, ...}, ...}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, metrics in results.items():
        row = {"Model": name}
        for k, v in metrics.items():
            if k != "residual_stats":
                row[k] = round(v, 4)
        rows.append(row)
    return pd.DataFrame(rows)
