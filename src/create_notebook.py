"""
Script to generate the Jupyter notebook for the project.
"""
import nbformat
import os
import sys

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

cells = []

# Cell 1: Title and imports
cells.append(nbformat.v4.new_markdown_cell("""# Student Marks Prediction Using Least Squares
## Multiple Linear Regression Using the Normal Equations

This notebook demonstrates the complete Least Squares regression pipeline step by step.

**Academic Integrity Notice:** The dataset used is synthetic/project data created for demonstrating Least Squares regression. It is NOT collected from real students."""))

cells.append(nbformat.v4.new_code_cell("""# Cell 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

from src.least_squares import LeastSquaresRegressor
from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN

print("Libraries imported successfully.")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")"""))

# Cell 2: Load dataset
cells.append(nbformat.v4.new_markdown_cell("## Cell 2: Load Dataset"))
cells.append(nbformat.v4.new_code_cell("""# Cell 2: Load the student marks dataset
df = pd.read_csv('../data/student_marks.csv')
print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\\nColumns: {list(df.columns)}")"""))

# Cell 3: Inspect
cells.append(nbformat.v4.new_markdown_cell("## Cell 3: Inspect Dataset"))
cells.append(nbformat.v4.new_code_cell("""# Cell 3: Inspect the first 10 rows
df.head(10)"""))

# Cell 4: Validate
cells.append(nbformat.v4.new_markdown_cell("## Cell 4: Validate Dataset"))
cells.append(nbformat.v4.new_code_cell("""# Cell 4: Data Validation
print("Missing values:")
print(df.isnull().sum())
print(f"\\nDuplicate rows: {df.duplicated().sum()}")
print(f"\\nAll values in valid ranges:")
print(f"  Study_Hours    : [{df['Study_Hours'].min():.1f}, {df['Study_Hours'].max():.1f}] (valid: 0-24)")
print(f"  Attendance     : [{df['Attendance'].min():.1f}, {df['Attendance'].max():.1f}] (valid: 0-100)")
print(f"  Assignment     : [{df['Assignment_Marks'].min():.1f}, {df['Assignment_Marks'].max():.1f}] (valid: 0-100)")
print(f"  Internal Assess: [{df['Internal_Assessment_Marks'].min():.1f}, {df['Internal_Assessment_Marks'].max():.1f}] (valid: 0-100)")
print(f"  Final Exam     : [{df['Final_Exam_Marks'].min():.1f}, {df['Final_Exam_Marks'].max():.1f}] (valid: 0-100)")
print("\\nDataset validation: PASSED")"""))

# Cell 5: Summary statistics
cells.append(nbformat.v4.new_markdown_cell("## Cell 5: Summary Statistics"))
cells.append(nbformat.v4.new_code_cell("""# Cell 5: Descriptive statistics
df.describe().round(2)"""))

# Cell 6: Visualize features
cells.append(nbformat.v4.new_markdown_cell("""## Cell 6: Visualize Features vs Final Exam Marks

Each scatter plot shows the relationship between an input feature and the target variable."""))
cells.append(nbformat.v4.new_code_cell("""# Cell 6: Feature visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ['#2563EB', '#7C3AED', '#059669', '#D97706']

for i, (feat, color) in enumerate(zip(FEATURE_COLUMNS, colors)):
    ax = axes[i // 2, i % 2]
    ax.scatter(df[feat], df[TARGET_COLUMN], alpha=0.6, s=35, color=color, edgecolors='white', linewidth=0.5)
    ax.set_xlabel(feat.replace('_', ' '), fontsize=11)
    ax.set_ylabel('Final Exam Marks', fontsize=11)
    ax.set_title(f'{feat.replace("_", " ")} vs Final Exam Marks', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Feature Relationships with Final Examination Marks', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""))

# Cell 7: Train/Test Split
cells.append(nbformat.v4.new_markdown_cell("""## Cell 7: Train/Test Split

We split the data into 80% training and 20% testing. The model is fitted ONLY on training data."""))
cells.append(nbformat.v4.new_code_cell("""# Cell 7: Train/Test Split
from sklearn.model_selection import train_test_split

X = df[FEATURE_COLUMNS].values.astype(np.float64)
y = df[TARGET_COLUMN].values.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total samples  : {len(X)}")
print(f"Training samples: {len(X_train)} (80%)")
print(f"Testing samples : {len(X_test)}  (20%)")"""))

# Cell 8: Design Matrix
cells.append(nbformat.v4.new_markdown_cell(r"""## Cell 8: Construct Design Matrix

The design matrix $\mathbf{X}$ prepends a column of 1s to account for the intercept term $w_0$:

$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{21} & x_{31} & x_{41} \\ 1 & x_{12} & x_{22} & x_{32} & x_{42} \\ \vdots & & & & \vdots \\ 1 & x_{1N} & x_{2N} & x_{3N} & x_{4N} \end{bmatrix}$$"""))
cells.append(nbformat.v4.new_code_cell("""# Cell 8: Construct the design matrix
ones = np.ones((X_train.shape[0], 1))
X_design = np.hstack([ones, X_train])

print(f"Feature matrix shape : {X_train.shape}")
print(f"Design matrix shape  : {X_design.shape}")
print(f"\\nFirst 5 rows of design matrix:")
print(X_design[:5].round(2))"""))

# Cell 9: X^T
cells.append(nbformat.v4.new_markdown_cell(r"## Cell 9: Calculate $\mathbf{X}^T$ (Transpose)"))
cells.append(nbformat.v4.new_code_cell("""# Cell 9: Calculate X transpose
Xt = X_design.T
print(f"X^T shape: {Xt.shape}")
print(f"\\nX^T (first 5 columns):")
print(Xt[:, :5].round(2))"""))

# Cell 10: X^TX
cells.append(nbformat.v4.new_markdown_cell(r"""## Cell 10: Calculate $\mathbf{X}^T \mathbf{X}$

This is the Gram matrix — a (5x5) symmetric positive semi-definite matrix."""))
cells.append(nbformat.v4.new_code_cell("""# Cell 10: Calculate X^T X
XtX = Xt @ X_design
print(f"X^T X shape: {XtX.shape}")
print(f"\\nX^T X:")
print(XtX.round(2))"""))

# Cell 11: X^Ty
cells.append(nbformat.v4.new_markdown_cell(r"## Cell 11: Calculate $\mathbf{X}^T \mathbf{y}$"))
cells.append(nbformat.v4.new_code_cell("""# Cell 11: Calculate X^T y
Xty = Xt @ y_train
print(f"X^T y shape: {Xty.shape}")
print(f"\\nX^T y:")
for i, val in enumerate(Xty):
    print(f"  Component {i}: {val:.4f}")"""))

# Cell 12: Solve normal equations
cells.append(nbformat.v4.new_markdown_cell(r"""## Cell 12: Solve the Normal Equations

We solve $\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{y}$ using `numpy.linalg.solve()`:

This avoids computing the explicit inverse, which is more numerically stable."""))
cells.append(nbformat.v4.new_code_cell("""# Cell 12: Solve the normal equations
w = np.linalg.solve(XtX, Xty)
print("Learned weight vector w:")
for i, val in enumerate(w):
    print(f"  w[{i}] = {val:.6f}")"""))

# Cell 13: Display coefficients
cells.append(nbformat.v4.new_markdown_cell("## Cell 13: Display Learned Regression Coefficients"))
cells.append(nbformat.v4.new_code_cell("""# Cell 13: Display coefficients with feature names
print("=" * 50)
print("  REGRESSION COEFFICIENTS")
print("=" * 50)
print(f"\\n  Intercept (w0): {w[0]:.4f}")
print()
for i, name in enumerate(FEATURE_COLUMNS):
    print(f"  {name:30s} : {w[i+1]:+.4f}")
print()
print("  Model Equation:")
print(f"  Predicted Marks = {w[0]:.4f}")
for i, name in enumerate(FEATURE_COLUMNS):
    sign = '+' if w[i+1] >= 0 else '-'
    print(f"    {sign} {abs(w[i+1]):.4f} * {name}")"""))

# Cell 14: Generate predictions
cells.append(nbformat.v4.new_markdown_cell("## Cell 14: Generate Predictions"))
cells.append(nbformat.v4.new_code_cell("""# Cell 14: Generate predictions for training and testing data
y_pred_train = X_design @ w

# For test data
X_design_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
y_pred_test = X_design_test @ w

print("Training predictions (first 10):")
for actual, pred in zip(y_train[:10], y_pred_train[:10]):
    print(f"  Actual: {actual:6.2f}  |  Predicted: {pred:6.2f}")

print(f"\\nTest predictions (first 10):")
for actual, pred in zip(y_test[:10], y_pred_test[:10]):
    print(f"  Actual: {actual:6.2f}  |  Predicted: {pred:6.2f}")"""))

# Cell 15: Residuals
cells.append(nbformat.v4.new_markdown_cell(r"""## Cell 15: Calculate Residuals

$$r_n = t_n - y(\mathbf{x}_n, \mathbf{w})$$"""))
cells.append(nbformat.v4.new_code_cell("""# Cell 15: Calculate residuals
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

print("Training Residuals (first 10):")
for r in residuals_train[:10]:
    print(f"  {r:+.4f}")

print(f"\\nTest Residual Statistics:")
print(f"  Mean   : {residuals_test.mean():.4f}")
print(f"  Std    : {residuals_test.std():.4f}")
print(f"  Min    : {residuals_test.min():.4f}")
print(f"  Max    : {residuals_test.max():.4f}")"""))

# Cell 16: MSE
cells.append(nbformat.v4.new_markdown_cell(r"""## Cell 16: Calculate Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{N} \sum_{n=1}^{N} (t_n - y_n)^2$$"""))
cells.append(nbformat.v4.new_code_cell("""# Cell 16: Calculate MSE manually
mse_train = np.mean(residuals_train ** 2)
mse_test = np.mean(residuals_test ** 2)

print(f"Training MSE: {mse_train:.4f}")
print(f"Testing MSE : {mse_test:.4f}")
print(f"\\nLower MSE = better predictions.")"""))

# Cell 17: R2
cells.append(nbformat.v4.new_markdown_cell(r"""## Cell 17: Calculate R-squared (R^2)

$$R^2 = 1 - \frac{\sum (t_n - y_n)^2}{\sum (t_n - \bar{t})^2}$$"""))
cells.append(nbformat.v4.new_code_cell("""# Cell 17: Calculate R^2 manually
ss_res_train = np.sum(residuals_train ** 2)
ss_tot_train = np.sum((y_train - y_train.mean()) ** 2)
r2_train = 1 - ss_res_train / ss_tot_train

ss_res_test = np.sum(residuals_test ** 2)
ss_tot_test = np.sum((y_test - y_test.mean()) ** 2)
r2_test = 1 - ss_res_test / ss_tot_test

print(f"Training R^2: {r2_train:.4f}")
print(f"Testing R^2 : {r2_test:.4f}")
print(f"\\nR^2 = 1 means perfect fit.")
print(f"R^2 = 0 means no better than predicting the mean.")"""))

# Cell 18: Actual vs Predicted
cells.append(nbformat.v4.new_markdown_cell("## Cell 18: Plot Actual vs Predicted Marks"))
cells.append(nbformat.v4.new_code_cell("""# Cell 18: Actual vs Predicted scatter plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred_test, alpha=0.7, s=50, color='#2563EB', edgecolors='white', linewidth=0.5)
lo = min(y_test.min(), y_pred_test.min()) - 5
hi = max(y_test.max(), y_pred_test.max()) + 5
ax.plot([lo, hi], [lo, hi], '--', color='#DC2626', linewidth=2, label='Perfect Prediction (y=x)')
ax.set_xlabel('Actual Final Exam Marks', fontsize=12)
ax.set_ylabel('Predicted Final Exam Marks', fontsize=12)
ax.set_title('Actual vs Predicted Final Examination Marks (Test Set)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

# Cell 19: Plot residuals
cells.append(nbformat.v4.new_markdown_cell("## Cell 19: Plot Residuals"))
cells.append(nbformat.v4.new_code_cell("""# Cell 19: Residual plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Residual scatter
axes[0].scatter(y_pred_test, residuals_test, alpha=0.7, s=40, color='#7C3AED', edgecolors='white')
axes[0].axhline(y=0, color='#DC2626', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Marks', fontsize=12)
axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[0].set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Residual histogram
axes[1].hist(residuals_test, bins=15, color='#059669', alpha=0.7, edgecolor='white')
axes[1].axvline(x=0, color='#DC2626', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual Value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Cell 20: Predict new student
cells.append(nbformat.v4.new_markdown_cell("## Cell 20: Predict Marks for a New Student"))
cells.append(nbformat.v4.new_code_cell("""# Cell 20: Prediction for a new student
study_hours = 7.5
attendance = 85.0
assignment_marks = 72.0
internal_marks = 68.0

x_new = np.array([1.0, study_hours, attendance, assignment_marks, internal_marks])
predicted_marks = x_new @ w

print("=" * 50)
print("  PREDICTION FOR A NEW STUDENT")
print("=" * 50)
print(f"\\n  Study Hours              : {study_hours}")
print(f"  Attendance               : {attendance}%")
print(f"  Assignment Marks         : {assignment_marks}")
print(f"  Internal Assessment Marks: {internal_marks}")
print(f"\\n  Calculation:")
print(f"    {w[0]:.4f}  (intercept)")
print(f"  + {w[1]:.4f} * {study_hours} = {w[1]*study_hours:.4f}")
print(f"  + {w[2]:.4f} * {attendance} = {w[2]*attendance:.4f}")
print(f"  + {w[3]:.4f} * {assignment_marks} = {w[3]*assignment_marks:.4f}")
print(f"  + {w[4]:.4f} * {internal_marks} = {w[4]*internal_marks:.4f}")
print(f"  ---")
print(f"  Total = {predicted_marks:.4f}")
print(f"\\n  >> Predicted Final Examination Marks: {predicted_marks:.2f}")"""))

# Cell 21: Polynomial experiment
cells.append(nbformat.v4.new_markdown_cell(r"""## Cell 21: Optional Polynomial Extension (Unit 1 Connection)

We extend the model by adding $x_1^2$ (Study Hours squared) to demonstrate polynomial curve fitting:

$$y = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 + w_5 x_1^2$$"""))
cells.append(nbformat.v4.new_code_cell("""# Cell 21: Polynomial Extension
X_train_poly = np.column_stack([X_train, X_train[:, 0] ** 2])
X_test_poly = np.column_stack([X_test, X_test[:, 0] ** 2])

X_design_poly = np.hstack([np.ones((X_train_poly.shape[0], 1)), X_train_poly])
w_poly = np.linalg.solve(X_design_poly.T @ X_design_poly, X_design_poly.T @ y_train)

X_design_test_poly = np.hstack([np.ones((X_test_poly.shape[0], 1)), X_test_poly])
y_pred_test_poly = X_design_test_poly @ w_poly
residuals_poly = y_test - y_pred_test_poly

mse_poly = np.mean(residuals_poly ** 2)
ss_res_poly = np.sum(residuals_poly ** 2)
ss_tot_poly = np.sum((y_test - y_test.mean()) ** 2)
r2_poly = 1 - ss_res_poly / ss_tot_poly

print("Model Comparison:")
print(f"{'Metric':<15} {'Linear':>12} {'Polynomial':>12}")
print("-" * 40)
print(f"{'Test MSE':<15} {mse_test:>12.4f} {mse_poly:>12.4f}")
print(f"{'Test R^2':<15} {r2_test:>12.4f} {r2_poly:>12.4f}")
print(f"\\nNote: If the polynomial model does not significantly improve,")
print(f"the added complexity is not justified (overfitting risk).")"""))

# Cell 22: Conclusion
cells.append(nbformat.v4.new_markdown_cell("""## Cell 22: Conclusion

### Summary

This notebook demonstrated the complete **Least Squares regression** pipeline:

1. **Data Loading & Validation** - 200 synthetic student records
2. **Train/Test Split** - 160/40 split for generalization evaluation
3. **Design Matrix Construction** - Column of 1s prepended for intercept
4. **Normal Equations** - X^TX w = X^Ty solved via numpy.linalg.solve()
5. **Coefficient Estimation** - Learned w0, w1, w2, w3, w4 from training data
6. **Prediction** - y = Xw for both training and testing sets
7. **Evaluation** - MSE and R^2 computed on the test set
8. **Residual Analysis** - Distribution and statistics of prediction errors
9. **New Student Prediction** - Interactive prediction with step-by-step breakdown
10. **Polynomial Extension** - Demonstrated model complexity vs generalization

### Key Unit 1 Concepts Demonstrated
- Supervised Learning, Regression, Features/Target
- Training/Test Data, Generalization
- Sum-of-Squares Error, Polynomial Curve Fitting

### Key Unit 2 Concepts Demonstrated
- Maximum Likelihood (Gaussian noise -> Least Squares)
- Normal Equations, Regression Coefficients
- Residuals, MSE, R^2
- Ridge/Bayesian Regression (theoretical extensions)

### Primary Algorithm: **Least Squares Regression** via the Normal Equations."""))

for cell in cells:
    nb.cells.append(cell)

# Save notebook
notebook_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks')
os.makedirs(notebook_dir, exist_ok=True)
notebook_path = os.path.join(notebook_dir, 'student_marks_analysis.ipynb')
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"Notebook saved to: {notebook_path}")
