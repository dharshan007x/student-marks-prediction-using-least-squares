"""
visualizations.py
=================
Plotting utilities for the Student Marks Prediction project.

Creates publication-quality academic visualizations using Matplotlib.
All figures are designed for clarity and professional presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import Optional, List, Tuple

# Use a non-interactive backend when saving to files
matplotlib.use("Agg")

# Consistent style
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "neutral": "#6B7280",
    "bg": "#F8FAFC",
}


def plot_feature_vs_target(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str = "Final_Exam_Marks",
    save_path: Optional[str] = None,
    color: str = COLORS["primary"],
) -> plt.Figure:
    """
    Scatter plot of a single feature vs the target variable.

    Parameters
    ----------
    df : pd.DataFrame
    feature_col : str
    target_col : str
    save_path : str or None
    color : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[feature_col], df[target_col], alpha=0.6, s=40,
               color=color, edgecolors="white", linewidth=0.5)
    ax.set_xlabel(feature_col.replace("_", " "))
    ax.set_ylabel(target_col.replace("_", " "))
    ax.set_title(f"{feature_col.replace('_', ' ')} vs {target_col.replace('_', ' ')}")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all_features_vs_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Final_Exam_Marks",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a 2×2 grid of scatter plots, one per feature.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str
    target_col : str
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors = [COLORS["primary"], COLORS["secondary"],
              COLORS["accent"], COLORS["warning"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (col, color) in enumerate(zip(feature_cols, colors)):
        ax = axes[i]
        ax.scatter(df[col], df[target_col], alpha=0.6, s=35,
                   color=color, edgecolors="white", linewidth=0.5)
        ax.set_xlabel(col.replace("_", " "), fontsize=11)
        ax.set_ylabel(target_col.replace("_", " "), fontsize=11)
        ax.set_title(f"{col.replace('_', ' ')} vs {target_col.replace('_', ' ')}",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feature Relationships with Final Examination Marks",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Final Examination Marks",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of actual vs predicted values with y=x reference line.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    title : str
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.6, s=50,
               color=COLORS["primary"], edgecolors="white",
               linewidth=0.5, label="Predictions", zorder=3)

    # Reference line y = x
    lo = min(np.min(y_true), np.min(y_pred)) - 5
    hi = max(np.max(y_true), np.max(y_pred)) + 5
    ax.plot([lo, hi], [lo, hi], "--", color=COLORS["danger"],
            linewidth=2, label="Perfect Prediction (y = x)", zorder=2)

    ax.set_xlabel("Actual Final Examination Marks", fontsize=12)
    ax.set_ylabel("Predicted Final Examination Marks", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Residual plot: predicted values vs residuals.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    title : str
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Residual scatter
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.6, s=40,
                color=COLORS["secondary"], edgecolors="white", linewidth=0.5)
    ax1.axhline(y=0, color=COLORS["danger"], linestyle="--", linewidth=2)
    ax1.set_xlabel("Predicted Final Marks", fontsize=12)
    ax1.set_ylabel("Residual (Actual − Predicted)", fontsize=12)
    ax1.set_title("Residuals vs Predicted Values", fontsize=13, fontweight="bold")

    # Right: Residual histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=20, color=COLORS["accent"], alpha=0.7,
             edgecolor="white", linewidth=0.8)
    ax2.axvline(x=0, color=COLORS["danger"], linestyle="--", linewidth=2)
    ax2.set_xlabel("Residual Value", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Residual Distribution", fontsize=13, fontweight="bold")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Correlation heatmap for the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str or None
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, shrink=0.8)

    labels = [c.replace("_", "\n") for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="left")
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")

    ax.set_title("Feature Correlation Matrix", fontsize=14,
                 fontweight="bold", pad=50)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_model_statistics(
    train_metrics: dict,
    test_metrics: dict,
    coefficients: dict,
    save_path: str,
) -> None:
    """
    Save model statistics to a CSV file.

    Parameters
    ----------
    train_metrics : dict
    test_metrics : dict
    coefficients : dict
    save_path : str
    """
    rows = []
    rows.append({"Category": "Training", "Metric": "MSE", "Value": train_metrics.get("MSE", "")})
    rows.append({"Category": "Training", "Metric": "R2", "Value": train_metrics.get("R2", "")})
    rows.append({"Category": "Training", "Metric": "MAE", "Value": train_metrics.get("MAE", "")})
    rows.append({"Category": "Training", "Metric": "RMSE", "Value": train_metrics.get("RMSE", "")})
    rows.append({"Category": "Testing", "Metric": "MSE", "Value": test_metrics.get("MSE", "")})
    rows.append({"Category": "Testing", "Metric": "R2", "Value": test_metrics.get("R2", "")})
    rows.append({"Category": "Testing", "Metric": "MAE", "Value": test_metrics.get("MAE", "")})
    rows.append({"Category": "Testing", "Metric": "RMSE", "Value": test_metrics.get("RMSE", "")})

    for name, val in coefficients.items():
        rows.append({"Category": "Coefficient", "Metric": name, "Value": val})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
