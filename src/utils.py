"""
utils.py
========
General utility functions for the Student Marks Prediction project.

Provides formatting helpers, path resolution, and display utilities.
"""

import os
import sys
import numpy as np
from typing import List, Optional


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename: str = "student_marks.csv") -> str:
    """Return the absolute path to a file in the data/ directory."""
    return os.path.join(get_project_root(), "data", filename)


def get_output_path(filename: str) -> str:
    """Return the absolute path to a file in the outputs/ directory."""
    path = os.path.join(get_project_root(), "outputs", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def print_header(title: str, char: str = "=", width: int = 60) -> None:
    """Print a formatted section header."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def print_subheader(title: str, char: str = "-", width: int = 50) -> None:
    """Print a formatted sub-section header."""
    print()
    print(f"  {title}")
    print(f"  {char * min(len(title), width)}")


def format_equation(
    intercept: float,
    coefficients: np.ndarray,
    feature_names: List[str],
    target_name: str = "Predicted Final Marks",
) -> str:
    """
    Format the regression equation as a multi-line string.

    Parameters
    ----------
    intercept : float
    coefficients : np.ndarray
    feature_names : list of str
    target_name : str

    Returns
    -------
    str
    """
    lines = [f"{target_name}"]
    lines.append(f"  = {intercept:.4f}")
    for name, coef in zip(feature_names, coefficients):
        sign = "+" if coef >= 0 else "-"
        lines.append(f"  {sign} {abs(coef):.4f} * {name}")
    return "\n".join(lines)


def format_matrix(
    matrix: np.ndarray,
    name: str = "Matrix",
    max_rows: int = 6,
    decimals: int = 2,
) -> str:
    """
    Format a NumPy matrix for console display.

    Parameters
    ----------
    matrix : np.ndarray
    name : str
    max_rows : int
    decimals : int

    Returns
    -------
    str
    """
    lines = [f"{name}  (shape: {matrix.shape})"]
    n_rows = matrix.shape[0]
    show = min(n_rows, max_rows)

    for i in range(show):
        row_str = "  [" + "  ".join(f"{v:>10.{decimals}f}" for v in matrix[i]) + "]"
        lines.append(row_str)

    if n_rows > max_rows:
        lines.append(f"  ... ({n_rows - max_rows} more rows)")

    return "\n".join(lines)


def format_vector(
    vector: np.ndarray,
    name: str = "Vector",
    decimals: int = 4,
) -> str:
    """
    Format a NumPy vector for console display.

    Parameters
    ----------
    vector : np.ndarray
    name : str
    decimals : int

    Returns
    -------
    str
    """
    lines = [f"{name}  (length: {len(vector)})"]
    for i, v in enumerate(vector):
        lines.append(f"  w[{i}] = {v:.{decimals}f}")
    return "\n".join(lines)


def predict_single_student(
    model,
    study_hours: float,
    attendance: float,
    assignment_marks: float,
    internal_marks: float,
) -> dict:
    """
    Predict final examination marks for a single student and
    return a detailed breakdown.

    Parameters
    ----------
    model : LeastSquaresRegressor
    study_hours, attendance, assignment_marks, internal_marks : float

    Returns
    -------
    dict
        Keys: input_values, feature_vector, predicted_marks,
              contribution_breakdown
    """
    input_values = {
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Assignment_Marks": assignment_marks,
        "Internal_Assessment_Marks": internal_marks,
    }

    x = np.array([[study_hours, attendance, assignment_marks, internal_marks]])
    predicted = model.predict(x)[0]

    # Contribution breakdown
    contributions = {"Intercept (w0)": model.intercept_}
    feature_names = model.feature_names_ or list(input_values.keys())
    feature_values = [study_hours, attendance, assignment_marks, internal_marks]
    for name, val, coef in zip(feature_names, feature_values, model.coefficients_):
        contributions[name] = coef * val

    return {
        "input_values": input_values,
        "feature_vector": x.flatten(),
        "predicted_marks": float(predicted),
        "contribution_breakdown": contributions,
    }
