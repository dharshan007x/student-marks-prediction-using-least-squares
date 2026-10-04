"""
data_loader.py
==============
Handles dataset loading, validation, and train/test splitting for the
Student Marks Prediction project.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import os


# Required columns and their valid ranges
FEATURE_COLUMNS = [
    "Study_Hours",
    "Attendance",
    "Assignment_Marks",
    "Internal_Assessment_Marks",
]

TARGET_COLUMN = "Final_Exam_Marks"

VALID_RANGES = {
    "Study_Hours": (0.0, 24.0),
    "Attendance": (0.0, 100.0),
    "Assignment_Marks": (0.0, 100.0),
    "Internal_Assessment_Marks": (0.0, 100.0),
    "Final_Exam_Marks": (0.0, 100.0),
}


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the student marks dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate the dataset and return a detailed report.

    Checks for:
        - Missing values
        - Duplicate rows
        - Invalid numeric values (NaN / Inf)
        - Values outside the valid range for each column

    Parameters
    ----------
    df : pd.DataFrame
        The student marks dataset.

    Returns
    -------
    dict
        Validation report with keys:
        'passed', 'total_rows', 'missing_values', 'duplicates',
        'invalid_numeric', 'out_of_range', 'issues'
    """
    report: Dict[str, any] = {
        "passed": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "duplicates": 0,
        "invalid_numeric": {},
        "out_of_range": {},
        "issues": [],
    }

    # ---- Missing values ----
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        n_missing = int(df[col].isna().sum())
        report["missing_values"][col] = n_missing
        if n_missing > 0:
            report["passed"] = False
            report["issues"].append(f"{col}: {n_missing} missing value(s)")

    # ---- Duplicate rows ----
    n_dup = int(df.duplicated().sum())
    report["duplicates"] = n_dup
    if n_dup > 0:
        report["issues"].append(f"{n_dup} duplicate row(s) found")

    # ---- Invalid numeric (Inf) ----
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            n_inf = int(np.isinf(df[col]).sum())
            report["invalid_numeric"][col] = n_inf
            if n_inf > 0:
                report["passed"] = False
                report["issues"].append(f"{col}: {n_inf} infinite value(s)")

    # ---- Range validation ----
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            below = int((df[col] < lo).sum())
            above = int((df[col] > hi).sum())
            total_bad = below + above
            report["out_of_range"][col] = {
                "below_min": below,
                "above_max": above,
                "total": total_bad,
                "valid_range": (lo, hi),
            }
            if total_bad > 0:
                report["passed"] = False
                report["issues"].append(
                    f"{col}: {total_bad} value(s) outside [{lo}, {hi}]"
                )

    return report


def format_validation_report(report: Dict) -> str:
    """Return a human-readable string of the validation report."""
    lines = []
    lines.append("=" * 55)
    lines.append("        DATASET VALIDATION REPORT")
    lines.append("=" * 55)
    lines.append(f"Total Rows    : {report['total_rows']}")
    lines.append(f"Total Columns : {report['total_columns']}")
    lines.append(f"Duplicates    : {report['duplicates']}")
    lines.append("")
    lines.append("Missing Values:")
    for col, n in report["missing_values"].items():
        status = "[OK]" if n == 0 else f"[FAIL] ({n})"
        lines.append(f"  {col:30s} {status}")
    lines.append("")
    lines.append("Range Validation:")
    for col, info in report["out_of_range"].items():
        lo, hi = info["valid_range"]
        bad = info["total"]
        status = "[OK]" if bad == 0 else f"[FAIL] ({bad} out of range)"
        lines.append(f"  {col:30s} [{lo:>5.1f}, {hi:>5.1f}]  {status}")
    lines.append("")
    status_str = "PASSED" if report["passed"] else "FAILED"
    lines.append(f"Overall Status: {status_str}")
    if report["issues"]:
        lines.append("\nIssues:")
        for issue in report["issues"]:
            lines.append(f"  - {issue}")
    lines.append("=" * 55)
    return "\n".join(lines)


def get_features_and_target(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target vector y from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with required columns.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target vector.
    """
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df[TARGET_COLUMN].values.astype(np.float64)
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float
        Fraction of data used for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Return a summary dictionary of the dataset.

    Returns
    -------
    dict
        Keys: n_rows, n_cols, features, target, stats
    """
    stats = df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().to_dict()
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "stats": stats,
    }
