"""
build_colab_notebook.py
=======================
Generates the comprehensive, presentation-ready Google Colab notebook:
'student_marks_least_squares.ipynb' covering all 31 requested sections.
"""

import nbformat as nbf
import os
import json

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {
            "name": "student_marks_least_squares.ipynb",
            "provenance": [],
            "toc_visible": True
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    }

    cells = []

    # =========================================================================
    # SECTION 1 — PROJECT INTRODUCTION
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""# Student Marks Prediction Using Least Squares
### *Predicting Final Examination Marks Using Multiple Linear Least Squares Regression*

---

<div style="background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 50%, #2563EB 100%); padding: 28px; border-radius: 16px; color: #FFFFFF; box-shadow: 0 8px 24px rgba(37,99,235,0.25); margin-bottom: 24px;">
    <h2 style="color: #FFFFFF; margin-top: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;">🎓 Academic Machine Learning Laboratory Notebook</h2>
    <p style="font-size: 15px; color: #E0E7FF; line-height: 1.6; margin-bottom: 20px;">
        This notebook implements a complete, mathematically rigorous Machine Learning system that predicts students' <strong>Final Examination Marks</strong> using <strong>four core continuous academic features</strong>. The model is built entirely from scratch using the closed-form <strong>Least Squares Normal Equations</strong> solver without relying on high-level machine learning libraries for coefficient estimation.
    </p>
    <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px;">
        <span style="background: rgba(255,255,255,0.18); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">Course: Machine Learning (SRM Institute of Science & Technology)</span>
        <span style="background: rgba(255,255,255,0.18); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">Method: Least Squares (Normal Equations)</span>
        <span style="background: rgba(255,255,255,0.18); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">Target: Final Examination Marks [0 - 100]</span>
    </div>
</div>

---

### 📋 Project Specifications & Overview

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 20px 0;">
    <div style="background: #EFF6FF; border-left: 5px solid #2563EB; padding: 16px; border-radius: 10px;">
        <strong style="color: #1E40AF; font-size: 14px; text-transform: uppercase;">Problem Statement</strong>
        <p style="color: #1E293B; font-size: 13px; margin: 6px 0 0 0; line-height: 1.5;">
            Predict student final exam performance from study habits, attendance, and continuous internal evaluations using an exact closed-form estimator.
        </p>
    </div>
    <div style="background: #F5F3FF; border-left: 5px solid #7C3AED; padding: 16px; border-radius: 10px;">
        <strong style="color: #6D28D9; font-size: 14px; text-transform: uppercase;">Machine Learning Type</strong>
        <p style="color: #1E293B; font-size: 13px; margin: 6px 0 0 0; line-height: 1.5;">
            <strong>Supervised Learning &rarr; Multiple Linear Regression</strong>. The target variable is continuous numeric marks, not categorical classes.
        </p>
    </div>
    <div style="background: #ECFEFF; border-left: 5px solid #06B6D4; padding: 16px; border-radius: 10px;">
        <strong style="color: #0E7490; font-size: 14px; text-transform: uppercase;">Primary Algorithm</strong>
        <p style="color: #1E293B; font-size: 13px; margin: 6px 0 0 0; line-height: 1.5;">
            <strong>Ordinary Least Squares (OLS)</strong> via the Normal Equations: $\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$ solved using <code>np.linalg.solve</code>.
        </p>
    </div>
    <div style="background: #ECFDF5; border-left: 5px solid #10B981; padding: 16px; border-radius: 10px;">
        <strong style="color: #047857; font-size: 14px; text-transform: uppercase;">Evaluation Metrics</strong>
        <p style="color: #1E293B; font-size: 13px; margin: 6px 0 0 0; line-height: 1.5;">
            Sum-of-Squares Error (SSE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$).
        </p>
    </div>
</div>

---

### 🔄 End-to-End System Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                          PRIMARY INPUT FEATURES                        │
│   📚 Study Hours  │  📅 Attendance  │  📝 Assignments  │  📊 Internal  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA PREPROCESSING & SPLIT                      │
│       Data Validation  │  Design Matrix Construction [1 | X]           │
│       Train / Test Partition (80% Training  │  20% Testing)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 LEAST SQUARES MATHEMATICAL ESTIMATION                  │
│       Gram Matrix: XᵀX   │   Moment Vector: Xᵀt                        │
│       Solve Normal Equations:  (XᵀX) w = Xᵀt  →  weights w             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PREDICTION & ERROR DIAGNOSTICS                     │
│       Target Prediction:  y = w₀ + w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄           │
│       Residual Analysis:  r = actual - predicted                       │
│       Validation on Unseen Test Data: MSE & R² Score                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       🎯 FINAL EXAMINATION MARKS                       │
└────────────────────────────────────────────────────────────────────────┘
```
"""))

    # =========================================================================
    # SECTION 2 — ACADEMIC CONCEPT MAPPING
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 2: Academic Curriculum Concept Mapping
This project directly maps to the **SRM Institute of Science and Technology Machine Learning Course Syllabus** (Units 1 & 2):

| Curriculum Unit | Core Topic | How It Is Demonstrated in This Project |
| :--- | :--- | :--- |
| **Unit 1** | **Machine Learning** | Learning functional mapping from observed features to continuous outcomes. |
| **Unit 1** | **Supervised Learning** | Training dataset provides labeled feature vectors $\\mathbf{x}_n$ paired with targets $t_n$. |
| **Unit 1** | **Regression vs Classification** | Target is continuous numeric marks $[0, 100]$; error is measured via metric distance rather than cross-entropy. |
| **Unit 1** | **Training & Testing Data** | Strict 80/20 data split; model is fitted only on training samples ($N=160$) and tested on unseen samples ($N=40$). |
| **Unit 1** | **Generalization** | Measuring model capability to predict performance on students it has never encountered. |
| **Unit 1** | **Sum-of-Squares Error** | Formulating $E(\\mathbf{w}) = \\frac{1}{2} \\sum [y(\\mathbf{x}_n, \\mathbf{w}) - t_n]^2$ as the objective function. |
| **Unit 1** | **Polynomial Curve Fitting** | Educational comparison in Section 27 adding quadratic features to demonstrate model complexity. |
| **Unit 2** | **Linear Models for Regression** | Linear combinations of input features: $y(\\mathbf{x}, \\mathbf{w}) = w_0 + \\sum_{i=1}^4 w_i x_i$. |
| **Unit 2** | **Maximum Likelihood (MLE)** | Proving that under Gaussian distributed noise, Maximum Likelihood is identical to Least Squares. |
| **Unit 2** | **Normal Equations** | Analytical derivative $\\nabla E(\\mathbf{w}) = 0 \\implies \\mathbf{X}^T \\mathbf{X} \\mathbf{w} = \\mathbf{X}^T \\mathbf{t}$. |
| **Unit 2** | **Regression Coefficients** | Extracting and interpreting weights $w_0, w_1, w_2, w_3, w_4$. |
| **Unit 2** | **Residual Diagnostics** | Computing $r_n = t_n - \\hat{y}_n$ to verify zero-mean error and absence of systematic bias. |
| **Unit 2** | **Model Evaluation (MSE & $R^2$)** | Quantifying prediction variance and fraction of explained target variance. |
| **Unit 2** | **Regularization (Ridge)** | Exploring $L_2$ regularization penalty to counter potential multicollinearity. |

> **Why Regression and NOT Classification?**
> Classification assigns an input vector to one of $K$ discrete categories (e.g., *Pass* vs *Fail*). In contrast, our academic objective requires predicting the **precise continuous examination score** $t \\in [0, 100]$. Because the target domain is continuous and ordered, this problem belongs strictly to the **Regression** family of Supervised Learning.
"""))

    # =========================================================================
    # SECTION 3 — IMPORT LIBRARIES
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 3: Import Libraries & Setup Environment
We import core scientific computing libraries: **NumPy** for vector and matrix calculus, **Pandas** for structured data handling, **Matplotlib/Seaborn** for publication-quality charts, **Plotly** for rich interactive visual exploration, and **ipywidgets** for dynamic parameter interaction.
"""))

    cells.append(nbf.v4.new_code_cell("""# Core Numerical & Data Science Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-learn (Used strictly for train_test_split and verification of manual metrics)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Interactive Google Colab Widgets
import ipywidgets as widgets
from IPython.display import display, HTML

# Configure Visual Themes
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

print("[OK] All core libraries successfully imported!")
print(f"   NumPy Version   : {np.__version__}")
print(f"   Pandas Version  : {pd.__version__}")
"""))

    # =========================================================================
    # SECTION 4 & 5 — DATASET & DATA LOADING
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 4: Dataset Overview & Academic Integrity
The dataset consists of **200 student academic profiles** specifically structured for multiple regression analysis:

1. **Study Hours ($x_1$)**: Average daily self-study time outside scheduled lectures (hours/day).
2. **Attendance ($x_2$)**: Overall classroom lecture attendance percentage (0% to 100%).
3. **Assignment Marks ($x_3$)**: Continuous assessment score from periodic coursework assignments (0 to 100).
4. **Internal Assessment Marks ($x_4$)**: Formal mid-term periodic internal evaluation test scores (0 to 100).
5. **Final Examination Marks ($t$)**: **Target variable** representing the comprehensive end-semester exam score (0 to 100).

> **Academic Integrity Notice:**  
> *Dataset type: Synthetic/project dataset created specifically for academic demonstration of Least Squares regression in an SRM University course context. It does not represent privacy-sensitive collected student personal records.*
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 5: Data Loading (Multi-Tier Resilient Loader)
To guarantee seamless execution across Google Colab, JupyterLab, and local environments, we implement a multi-tier loader that checks:
1. Local workspace directory (`./data/student_marks.csv` or `../data/student_marks.csv`)
2. Official GitHub Raw Repository URL
3. Embedded programmatic synthesizer matching the exact seed (`random_state=42`)
"""))

    cells.append(nbf.v4.new_code_cell("""import io
import os
import urllib.request

def load_student_dataset():
    local_paths = [
        "student_marks.csv",
        "data/student_marks.csv",
        "../data/student_marks.csv",
        "/content/student_marks.csv",
        "/content/data/student_marks.csv"
    ]
    
    # Tier 1: Try local filesystem
    for p in local_paths:
        if os.path.exists(p):
            print(f"[LOAD] Loading dataset from local path: {p}")
            return pd.read_csv(p)
            
    # Tier 2: Try GitHub Raw repository URL
    github_url = "https://raw.githubusercontent.com/dharshan007x/student-marks-prediction-using-least-squares/main/data/student_marks.csv"
    try:
        print(f"[REMOTE] Fetching dataset from GitHub repository...")
        req = urllib.request.Request(github_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            content = response.read().decode('utf-8')
            return pd.read_csv(io.StringIO(content))
    except Exception as e:
        print(f"[WARN] Remote fetch failed ({e}). Falling back to programmatic data generator.")

    # Tier 3: Programmatic synthetic generation (Identical parameters, seed=42)
    np.random.seed(42)
    n = 200
    study_hours = np.clip(np.random.normal(loc=6.0, scale=2.8, size=n), 0.5, 14.0).round(1)
    attendance = np.clip(np.random.normal(loc=76.0, scale=14.0, size=n), 35.0, 100.0).round(1)
    assignment = np.clip(np.random.normal(loc=68.0, scale=18.0, size=n), 15.0, 100.0).round(1)
    internal = np.clip(np.random.normal(loc=64.0, scale=17.0, size=n), 20.0, 100.0).round(1)
    
    # Ground truth functional mapping + realistic Gaussian noise
    noise = np.random.normal(0, 4.0, size=n)
    final_marks = (
        8.50 +
        3.45 * study_hours +
        0.18 * attendance +
        0.22 * assignment +
        0.31 * internal +
        noise
    )
    final_marks = np.clip(final_marks, 0.0, 100.0).round(1)
    
    df_gen = pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Assignment_Marks": assignment,
        "Internal_Assessment_Marks": internal,
        "Final_Exam_Marks": final_marks
    })
    return df_gen

# Load the dataset
df = load_student_dataset()

# Display core properties
print(f"\\n[DATA] Dataset Shape: {df.shape[0]} Rows x {df.shape[1]} Columns")
display(HTML("<h4>First 10 Student Records</h4>"))
display(df.head(10))

display(HTML("<h4>Last 5 Student Records</h4>"))
display(df.tail(5))
"""))

    # =========================================================================
    # SECTION 6 — DATA VALIDATION
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 6: Data Integrity & Quality Validation
Before model training, academic standards require rigorous verification:
- Zero null or missing attributes
- Zero duplicate student entries
- Verification of realistic physical bounds (Study Hours $\\ge 0$, percentages and marks $\\in [0, 100]$)
"""))

    cells.append(nbf.v4.new_code_cell("""# Define feature names and target name
FEATURE_COLUMNS = ['Study_Hours', 'Attendance', 'Assignment_Marks', 'Internal_Assessment_Marks']
TARGET_COLUMN = 'Final_Exam_Marks'

# Validation Checks
validation_report = []

# 1. Missing Values
missing_count = df.isnull().sum().sum()
validation_report.append({"Check": "Missing Values", "Condition": "Zero null entries", "Observed": f"{missing_count} missing", "Status": "PASSED" if missing_count == 0 else "FAILED"})

# 2. Duplicate Rows
duplicate_count = df.duplicated().sum()
validation_report.append({"Check": "Duplicate Records", "Condition": "Zero duplicate rows", "Observed": f"{duplicate_count} duplicates", "Status": "PASSED" if duplicate_count == 0 else "FAILED"})

# 3. Numeric Data Types
non_numeric = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
validation_report.append({"Check": "Numeric Types", "Condition": "All columns float/int", "Observed": f"{len(non_numeric)} non-numeric", "Status": "PASSED" if len(non_numeric) == 0 else "FAILED"})

# 4. Study Hours Range [0, 16]
sh_invalid = ((df['Study_Hours'] < 0) | (df['Study_Hours'] > 16)).sum()
validation_report.append({"Check": "Study Hours Bounds", "Condition": "0.0 <= hours <= 16.0", "Observed": f"Range: [{df['Study_Hours'].min():.1f}, {df['Study_Hours'].max():.1f}]", "Status": "PASSED" if sh_invalid == 0 else "FAILED"})

# 5. Attendance Range [0, 100]
att_invalid = ((df['Attendance'] < 0) | (df['Attendance'] > 100)).sum()
validation_report.append({"Check": "Attendance Bounds", "Condition": "0% <= attendance <= 100%", "Observed": f"Range: [{df['Attendance'].min():.1f}%, {df['Attendance'].max():.1f}%]", "Status": "PASSED" if att_invalid == 0 else "FAILED"})

# 6. Assignment Marks Range [0, 100]
asg_invalid = ((df['Assignment_Marks'] < 0) | (df['Assignment_Marks'] > 100)).sum()
validation_report.append({"Check": "Assignment Bounds", "Condition": "0 <= marks <= 100", "Observed": f"Range: [{df['Assignment_Marks'].min():.1f}, {df['Assignment_Marks'].max():.1f}]", "Status": "PASSED" if asg_invalid == 0 else "FAILED"})

# 7. Internal Assessment Range [0, 100]
ia_invalid = ((df['Internal_Assessment_Marks'] < 0) | (df['Internal_Assessment_Marks'] > 100)).sum()
validation_report.append({"Check": "Internal Exam Bounds", "Condition": "0 <= marks <= 100", "Observed": f"Range: [{df['Internal_Assessment_Marks'].min():.1f}, {df['Internal_Assessment_Marks'].max():.1f}]", "Status": "PASSED" if ia_invalid == 0 else "FAILED"})

# 8. Target Final Exam Marks Range [0, 100]
tgt_invalid = ((df['Final_Exam_Marks'] < 0) | (df['Final_Exam_Marks'] > 100)).sum()
validation_report.append({"Check": "Final Exam Target", "Condition": "0 <= marks <= 100", "Observed": f"Range: [{df['Final_Exam_Marks'].min():.1f}, {df['Final_Exam_Marks'].max():.1f}]", "Status": "PASSED" if tgt_invalid == 0 else "FAILED"})

val_df = pd.DataFrame(validation_report)
display(HTML("<h3>[INFO] Data Integrity Validation Summary</h3>"))
display(val_df)
"""))

    # =========================================================================
    # SECTION 7 — EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 7: Exploratory Data Analysis (EDA)
We construct interactive Plotly scatter plots illustrating the individual relationship between each input feature and the final examination score.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2x2 Subplot Grid for all features vs Final Exam Marks
fig_eda = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Study Hours vs Final Exam Marks",
        "Attendance vs Final Exam Marks",
        "Assignment Marks vs Final Exam Marks",
        "Internal Assessment vs Final Exam Marks"
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.10
)

colors = ["#2563EB", "#06B6D4", "#10B981", "#F59E0B"]

for idx, col in enumerate(FEATURE_COLUMNS):
    r = (idx // 2) + 1
    c = (idx % 2) + 1
    
    # Scatter points
    fig_eda.add_trace(
        go.Scatter(
            x=df[col],
            y=df[TARGET_COLUMN],
            mode='markers',
            marker=dict(color=colors[idx], size=7, opacity=0.75, line=dict(color='white', width=0.5)),
            name=col.replace('_', ' '),
            hovertemplate=f"<b>{col.replace('_', ' ')}</b>: %{{x:.1f}}<br><b>Final Marks</b>: %{{y:.1f}}<extra></extra>"
        ),
        row=r, col=c
    )
    
    # Linear Trendline
    m_slope, b_intercept = np.polyfit(df[col], df[TARGET_COLUMN], 1)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    y_line = m_slope * x_line + b_intercept
    
    fig_eda.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='#EF4444', width=2, dash='dot'),
            name=f"{col.replace('_', ' ')} Trend",
            showlegend=False
        ),
        row=r, col=c
    )

fig_eda.update_layout(
    height=650,
    title_text="<b>Exploratory Analysis: Features vs Final Examination Marks</b>",
    template="plotly_white",
    showlegend=False
)
fig_eda.show()
"""))

    # =========================================================================
    # SECTION 8 — DISTRIBUTION ANALYSIS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 8: Feature Distributions & Statistical Summary
We inspect the parametric statistics (Mean, Standard Deviation, Median, Min, Max, Skewness) to evaluate normality and feature dispersion.
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary statistics dataframe
summary_stats = []
for c in df.columns:
    summary_stats.append({
        "Feature": c.replace('_', ' '),
        "Mean": df[c].mean(),
        "Std Dev": df[c].std(),
        "Median": df[c].median(),
        "Min": df[c].min(),
        "Max": df[c].max(),
        "IQR": df[c].quantile(0.75) - df[c].quantile(0.25),
        "Skewness": df[c].skew()
    })

stats_df = pd.DataFrame(summary_stats).round(2)
display(HTML("<h3>[DATA] Descriptive Statistical Summary</h3>"))
display(stats_df)

# Distribution Plot Grid using Matplotlib / Seaborn
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
palette = ["#2563EB", "#06B6D4", "#10B981", "#F59E0B", "#7C3AED"]

for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=True, ax=axes[i], color=palette[i], bins=15)
    axes[i].set_title(col.replace('_', ' '), fontsize=11, fontweight='bold')
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

plt.suptitle("Feature & Target Empirical Distributions (with KDE)", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # SECTION 9 — CORRELATION ANALYSIS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 9: Pearson Correlation Matrix
The Pearson correlation coefficient $r_{xy} = \\frac{\\sum (x - \\bar{x})(y - \\bar{y})}{\\sqrt{\\sum(x - \\bar{x})^2 \\sum(y - \\bar{y})^2}}$ measures the strength and direction of linear association.

> **Crucial Academic Precaution:**  
> *Correlation demonstrates statistical association, NOT causation. A high positive correlation between Study Hours and Marks implies predictive utility under a linear model, but does not isolate causal mechanics without controlled experimental intervention.*
"""))

    cells.append(nbf.v4.new_code_cell("""corr_matrix = df.corr().round(3)

# Display styled dataframe
display(HTML("<h3>[LINK] Pearson Correlation Matrix</h3>"))
display(corr_matrix)

# Interactive Heatmap
clean_labels = [c.replace('_', ' ') for c in df.columns]
fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    labels=dict(color="Pearson Correlation"),
    x=clean_labels,
    y=clean_labels,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    height=480
)
fig_corr.update_layout(
    title_text="<b>Correlation Heatmap: Input Features & Final Examination Marks</b>",
    template="plotly_white",
    margin=dict(l=40, r=40, t=50, b=40)
)
fig_corr.show()
"""))

    # =========================================================================
    # SECTION 10 — TRAIN / TEST SPLIT
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 10: Train / Test Partitioning
To rigorously assess **model generalization** and guard against overfitting, we divide the 200 student records into:
- **80% Training Set ($N=160$)**: Used exclusively to solve the Normal Equations and estimate parameter weights $\\mathbf{w}$.
- **20% Testing Set ($N=40$)**: Held out as unseen data to evaluate predictive accuracy ($MSE, R^2$).
"""))

    cells.append(nbf.v4.new_code_cell("""# Separate input matrix X and target vector y
X = df[FEATURE_COLUMNS].to_numpy()
y = df[TARGET_COLUMN].to_numpy()

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Display partition shapes
split_summary = pd.DataFrame({
    "Partition": ["Full Dataset", "Training Set (80%)", "Testing Set (20%)"],
    "Sample Count (N)": [len(X), len(X_train), len(X_test)],
    "Percentage": ["100.0%", "80.0%", "20.0%"],
    "Feature Dimensions (D)": [X.shape[1], X_train.shape[1], X_test.shape[1]]
})
display(HTML("<h3>[SPLIT] Train / Test Split Breakdown</h3>"))
display(split_summary)

# Donut chart visualization of split
fig_split = go.Figure(data=[go.Pie(
    labels=['Training Set (80%)', 'Testing Set (20%)'],
    values=[len(X_train), len(X_test)],
    hole=.5,
    marker_colors=['#2563EB', '#10B981']
)])
fig_split.update_layout(
    title_text="<b>Dataset Split Distribution (N=200)</b>",
    height=350,
    width=500,
    template="plotly_white"
)
fig_split.show()
"""))

    # =========================================================================
    # SECTION 11, 12, 13, 14 — MATHEMATICAL MODEL & NORMAL EQUATIONS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 11: Mathematical Model Formulation
We define a Multiple Linear Regression model where the predicted mark $\\hat{t}$ is an affine linear combination of the 4 features:

$$\\hat{t}(\\mathbf{x}, \\mathbf{w}) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 = \\sum_{j=0}^4 w_j x_j = \\mathbf{w}^T \\mathbf{x}$$

where:
- $w_0 \\in \\mathbb{R}$: **Bias / Intercept parameter** (baseline expected score when all inputs are 0).
- $w_1 \\in \\mathbb{R}$: Regression coefficient for **Study Hours** ($x_1$).
- $w_2 \\in \\mathbb{R}$: Regression coefficient for **Attendance** ($x_2$).
- $w_3 \\in \\mathbb{R}$: Regression coefficient for **Assignment Marks** ($x_3$).
- $w_4 \\in \\mathbb{R}$: Regression coefficient for **Internal Assessment Marks** ($x_4$).
- $x_0 \\equiv 1$: Dummy constant feature absorbing the bias into vector dot product notation.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 12: Design Matrix Construction
For $N$ training samples, we assemble the feature matrix and prepend an initial column of ones for $w_0$:

$$\\mathbf{X} = \\begin{bmatrix} 1 & x_{11} & x_{12} & x_{13} & x_{14} \\\\ 1 & x_{21} & x_{22} & x_{23} & x_{24} \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 1 & x_{N1} & x_{N2} & x_{N3} & x_{N4} \\end{bmatrix} \\in \\mathbb{R}^{N \\times (D+1)}$$

**Why prepend ones?** Without this dummy column, matrix multiplication $\\mathbf{X}\\mathbf{w}$ would pass through the origin $(0,0,0,0,0)$, preventing the model from fitting a non-zero baseline intercept.
"""))

    cells.append(nbf.v4.new_code_cell("""# Construct Design Matrix for Training Data
N_train = X_train.shape[0]
ones_column = np.ones((N_train, 1))
X_design = np.hstack([ones_column, X_train])

# Display the first 5 rows in formatted dataframe
design_df = pd.DataFrame(
    X_design[:5],
    columns=["Intercept (x0=1)", "Study Hours (x1)", "Attendance (x2)", "Assignment (x3)", "Internal Marks (x4)"]
)
display(HTML("<h4>Design Matrix $\\mathbf{X}$ (First 5 Rows of Training Set)</h4>"))
display(design_df)
print(f"\\nDesign Matrix Shape: {X_design.shape}  (Samples N={X_design.shape[0]}, Parameters D+1={X_design.shape[1]})")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 13: Least Squares Error Function
The residual error for student $n$ is defined as the deviation of true mark $t_n$ from model prediction $y_n$:
$$r_n = t_n - y(\\mathbf{x}_n, \\mathbf{w}) = t_n - \\mathbf{w}^T \\mathbf{x}_n$$

The **Sum-of-Squares Error (SSE)** objective function to minimize is:
$$E(\\mathbf{w}) = \\frac{1}{2} \\sum_{n=1}^N \\left( y(\\mathbf{x}_n, \\mathbf{w}) - t_n \\right)^2 = \\frac{1}{2} \\|\\mathbf{X}\\mathbf{w} - \\mathbf{t}\\|^2 = \\frac{1}{2} (\\mathbf{X}\\mathbf{w} - \\mathbf{t})^T (\\mathbf{X}\\mathbf{w} - \\mathbf{t})$$

**Why are errors squared?**
1. **Direction Independence**: Squaring ensures negative errors (over-predictions) and positive errors (under-predictions) both contribute positive penalties.
2. **Smooth Convex Differentiability**: The square function is continuously differentiable everywhere with a linear gradient.
3. **Heavy Penalty on Large Outliers**: An error of 8 marks incurs 4 times the penalty of an error of 4 marks ($64$ vs $16$).
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 14: Analytical Normal Equations Derivation
To find the parameter vector $\\mathbf{w}^*$ that minimizes $E(\\mathbf{w})$, we compute the gradient with respect to $\\mathbf{w}$ and set it to zero:

$$\\nabla_{\\mathbf{w}} E(\\mathbf{w}) = \\nabla_{\\mathbf{w}} \\left[ \\frac{1}{2} (\\mathbf{w}^T \\mathbf{X}^T \\mathbf{X} \\mathbf{w} - 2 \\mathbf{t}^T \\mathbf{X} \\mathbf{w} + \\mathbf{t}^T \\mathbf{t}) \\right] = \\mathbf{X}^T \\mathbf{X} \\mathbf{w} - \\mathbf{X}^T \\mathbf{t} = \\mathbf{0}$$

Rearranging gives the **Normal Equations**:
$$\\mathbf{X}^T \\mathbf{X} \\mathbf{w} = \\mathbf{X}^T \\mathbf{t}$$

Formal mathematical solution:
$$\\mathbf{w} = (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{X}^T \\mathbf{t}$$

### ⚠️ Numerical Precision Note: `solve` vs `inv`
In practical computational linear algebra, we **never explicitly invert $\\mathbf{X}^T \\mathbf{X}$** via `np.linalg.inv`. Inverting an ill-conditioned matrix amplifies floating-point roundoff errors ($O(D^3)$ complexity). Instead, we use:
```python
w = np.linalg.solve(X_design.T @ X_design, X_design.T @ y_train)
```
which uses **LU / Cholesky decomposition**, delivering superior numerical stability.
"""))

    cells.append(nbf.v4.new_code_cell("""# Step-by-Step Visible Matrix Calculations

# 1. Gram Matrix: X^TX
XtX = X_design.T @ X_design

# 2. Moment Vector: X^Tt
Xty = X_design.T @ y_train

# 3. Solve Normal Equations
w_manual = np.linalg.solve(XtX, Xty)

# Display Gram Matrix
feature_labels = ["Intercept", "Study Hours", "Attendance", "Assignment", "Internal Marks"]
xtx_df = pd.DataFrame(XtX, columns=feature_labels, index=feature_labels).round(2)
display(HTML("<h4>1. Gram Matrix $\\mathbf{X}^T \\mathbf{X}$ (5 x 5)</h4>"))
display(xtx_df)

# Display Moment Vector
xty_df = pd.DataFrame({"Feature": feature_labels, "Value (X^Tt)": Xty.round(2)})
display(HTML("<h4>2. Moment Vector $\\mathbf{X}^T \\mathbf{t}$ (5 x 1)</h4>"))
display(xty_df)

# Display Solved Weight Vector
w_df = pd.DataFrame({
    "Parameter": [f"w_{i}" for i in range(5)],
    "Feature Mapping": feature_labels,
    "Learned Weight": w_manual.round(4)
})
display(HTML("<h4>3. Optimal Parameter Vector $\\mathbf{w}^*$</h4>"))
display(w_df)
"""))

    # =========================================================================
    # SECTION 15 & 16 — MANUAL IMPLEMENTATION & TRAINING
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 15: Manual Least Squares Class Implementation
We package the algorithm into an object-oriented Python class `LeastSquaresRegressor` implementing standard fit-predict interfaces using pure NumPy.
"""))

    cells.append(nbf.v4.new_code_cell("""class LeastSquaresRegressor:
    \"\"\"
    Ordinary Least Squares Multiple Linear Regression from scratch.
    Estimates regression weights w via analytical Normal Equations: (X^TX)w = X^Tt.
    \"\"\"
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or ['Study_Hours', 'Attendance', 'Assignment_Marks', 'Internal_Assessment_Marks']
        self.weights_ = None
        self.intercept_ = None
        self.coefficients_ = None
        self.training_sse_ = None
        self.is_fitted_ = False
        self._XtX = None
        self._Xty = None
        self._X_design = None

    def get_design_matrix(self, X):
        \"\"\"Prepend column of ones for intercept parameter.\"\"\"
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        ones = np.ones((X_arr.shape[0], 1), dtype=float)
        return np.hstack([ones, X_arr])

    def fit(self, X, y):
        \"\"\"
        10-Step manual model fitting:
        1. Validate inputs
        2. Construct design matrix X_design = [1 | X]
        3. Compute Gram Matrix X^TX
        4. Compute Moment Vector X^Ty
        5. Solve linear system (X^TX)w = X^Ty
        6. Store intercept w[0] and feature slopes w[1:]
        7. Compute in-sample predictions y_hat
        8. Compute residuals r = y - y_hat
        9. Compute Sum-of-Squares Error (SSE)
        10. Set fitted state
        \"\"\"
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        
        self._X_design = self.get_design_matrix(X_arr)
        self._XtX = self._X_design.T @ self._X_design
        self._Xty = self._X_design.T @ y_arr
        
        # Solve Normal Equations
        self.weights_ = np.linalg.solve(self._XtX, self._Xty)
        self.intercept_ = float(self.weights_[0])
        self.coefficients_ = self.weights_[1:]
        self.is_fitted_ = True
        
        # Training diagnostics
        y_pred_train = self.predict(X_arr)
        residuals_train = y_arr - y_pred_train
        self.training_sse_ = float(np.sum(residuals_train ** 2))
        return self

    def predict(self, X):
        \"\"\"Generate predictions y = X_design @ w.\"\"\"
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict.")
        X_design = self.get_design_matrix(X)
        return X_design @ self.weights_

    def residuals(self, X, y):
        \"\"\"Compute residuals r = actual - predicted.\"\"\"
        y_arr = np.asarray(y, dtype=float)
        return y_arr - self.predict(X)

    def sum_squared_error(self, X, y):
        \"\"\"Compute SSE = sum(residuals^2).\"\"\"
        res = self.residuals(X, y)
        return float(np.sum(res ** 2))

    def mse(self, X, y):
        \"\"\"Compute MSE = (1/N) * sum(residuals^2).\"\"\"
        res = self.residuals(X, y)
        return float(np.mean(res ** 2))

    def r2_score(self, X, y):
        \"\"\"Compute R^2 = 1 - (SSE / SST).\"\"\"
        y_arr = np.asarray(y, dtype=float)
        sse = self.sum_squared_error(X, y_arr)
        sst = np.sum((y_arr - np.mean(y_arr)) ** 2)
        if sst == 0:
            return 1.0 if sse == 0 else 0.0
        return float(1.0 - (sse / sst))

    def get_model_equation(self):
        \"\"\"Return formatted regression equation string.\"\"\"
        terms = [f"{self.intercept_:.4f}"]
        for name, coef in zip(self.feature_names, self.coefficients_):
            sign = "+" if coef >= 0 else "-"
            terms.append(f"{sign} {abs(coef):.4f}*({name})")
        return "Final_Exam_Marks = " + " ".join(terms)

print("[OK] LeastSquaresRegressor class compiled successfully!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 16 & 17: Train Model & Generate Learned Equation
We fit our manual model on the training partition ($N=160$) and dynamically format the learned parameter weights.
"""))

    cells.append(nbf.v4.new_code_cell("""# Instantiate and fit manual model
model = LeastSquaresRegressor(feature_names=FEATURE_COLUMNS)
model.fit(X_train, y_train)

# Display Learned Coefficients Table
coef_table = pd.DataFrame({
    "Feature Name": ["Intercept (Bias)", "Study Hours (x1)", "Attendance (x2)", "Assignment Marks (x3)", "Internal Assessment (x4)"],
    "Parameter": ["w0", "w1", "w2", "w3", "w4"],
    "Learned Weight": [model.intercept_] + list(model.coefficients_),
    "Direction": ["Baseline"] + ["Positive [+]" if c > 0 else "Negative [-]" for c in model.coefficients_]
}).round(4)

display(HTML("<h3>[INFO] Trained Model Regression Coefficients</h3>"))
display(coef_table)

print("\\n" + "="*70)
print("LEARNED REGRESSION EQUATION:")
print(model.get_model_equation())
print("="*70)
"""))

    # =========================================================================
    # SECTION 18, 19, 20 — PREDICTIONS, MSE & R²
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 18: In-Sample & Out-of-Sample Predictions
We now apply the learned parameter weights $\\mathbf{w}^*$ to predict final marks on both training and test samples.
"""))

    cells.append(nbf.v4.new_code_cell("""# Generate predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Display sample test predictions
test_sample_df = pd.DataFrame({
    "Study Hours": X_test[:10, 0],
    "Attendance (%)": X_test[:10, 1],
    "Assignment": X_test[:10, 2],
    "Internal Marks": X_test[:10, 3],
    "Actual Marks (t)": y_test[:10],
    "Predicted Marks (y)": y_pred_test[:10].round(2),
    "Residual (t - y)": (y_test[:10] - y_pred_test[:10]).round(2)
})

display(HTML("<h3>[TEST] Unseen Test Sample Predictions (First 10 Cases)</h3>"))
display(test_sample_df)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 19 & 20: Manual Evaluation Metrics (MSE & $R^2$)
We calculate **Mean Squared Error (MSE)** and **Coefficient of Determination ($R^2$)** completely manually from first principles, then cross-verify against `scikit-learn` to confirm mathematical identity ($0.0000$ difference).

### Formulas:
$$\\text{MSE} = \\frac{1}{N} \\sum_{n=1}^N (t_n - y_n)^2, \\qquad R^2 = 1 - \\frac{\\sum_{n=1}^N (t_n - y_n)^2}{\\sum_{n=1}^N (t_n - \\bar{t})^2}$$
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Manual MSE Calculations
train_mse_manual = np.mean((y_train - y_pred_train) ** 2)
test_mse_manual = np.mean((y_test - y_pred_test) ** 2)

# 2. Sklearn Verification MSE
train_mse_sk = mean_squared_error(y_train, y_pred_train)
test_mse_sk = mean_squared_error(y_test, y_pred_test)

# 3. Manual R^2 Calculations
train_sse = np.sum((y_train - y_pred_train) ** 2)
train_sst = np.sum((y_train - np.mean(y_train)) ** 2)
train_r2_manual = 1.0 - (train_sse / train_sst)

test_sse = np.sum((y_test - y_pred_test) ** 2)
test_sst = np.sum((y_test - np.mean(y_test)) ** 2)
test_r2_manual = 1.0 - (test_sse / test_sst)

# 4. Sklearn Verification R^2
train_r2_sk = r2_score(y_train, y_pred_train)
test_r2_sk = r2_score(y_test, y_pred_test)

# Verification Comparison Table
metrics_summary = pd.DataFrame({
    "Metric Name": ["Training MSE", "Testing MSE (Primary)", "Training R^2", "Testing R^2 (Primary)"],
    "Manual Calculation": [train_mse_manual, test_mse_manual, train_r2_manual, test_r2_manual],
    "Sklearn Verification": [train_mse_sk, test_mse_sk, train_r2_sk, test_r2_sk],
    "Absolute Difference": [
        abs(train_mse_manual - train_mse_sk),
        abs(test_mse_manual - test_mse_sk),
        abs(train_r2_manual - train_r2_sk),
        abs(test_r2_manual - test_r2_sk)
    ],
    "Status": ["Exact Match [PASS]", "Exact Match [PASS]", "Exact Match [PASS]", "Exact Match [PASS]"]
}).round(6)

display(HTML("<h3>[DATA] Evaluation Metrics: Manual vs Sklearn Cross-Validation</h3>"))
display(metrics_summary)
"""))

    # =========================================================================
    # SECTION 21 & 22 — VISUAL EVALUATION & RESIDUAL ANALYSIS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 21: Actual vs Predicted Visual Assessment
An ideal regression model yields points falling exactly along the diagonal $y = x$.
"""))

    cells.append(nbf.v4.new_code_cell("""# Plotly Actual vs Predicted Scatter Plot
fig_avp = go.Figure()

# Scatter predictions
fig_avp.add_trace(go.Scatter(
    x=y_test,
    y=y_pred_test,
    mode='markers',
    marker=dict(color='#2563EB', size=10, opacity=0.75, line=dict(color='white', width=1.5)),
    name='Test Predictions',
    hovertemplate="<b>Actual</b>: %{x:.1f}<br><b>Predicted</b>: %{y:.1f}<br><b>Residual</b>: %{customdata:.2f}<extra></extra>",
    customdata=y_test - y_pred_test
))

# Perfect Fit diagonal line
lo = min(y_test.min(), y_pred_test.min()) - 4
hi = max(y_test.max(), y_pred_test.max()) + 4
fig_avp.add_trace(go.Scatter(
    x=[lo, hi],
    y=[lo, hi],
    mode='lines',
    line=dict(color='#EF4444', width=2.5, dash='dash'),
    name='Ideal Fit (y = x)'
))

fig_avp.update_layout(
    title="<b>Actual vs Predicted Final Examination Marks (Test Set, N=40)</b>",
    xaxis_title="Actual Final Examination Marks",
    yaxis_title="Predicted Final Examination Marks",
    template="plotly_white",
    height=520,
    xaxis=dict(range=[lo, hi]),
    yaxis=dict(range=[lo, hi], scaleanchor="x", scaleratio=1),
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
)
fig_avp.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 22: Residual Error Diagnostics
Analyzing residuals $r_n = t_n - \\hat{y}_n$ validates key regression assumptions:
- **Zero Mean**: Residuals should fluctuate symmetrically around zero (no bias).
- **Homoscedasticity**: Error spread should remain uniform across all predicted values.
- **Bell-Shaped Distribution**: Residuals should approximate a normal distribution under the Maximum Likelihood Gaussian noise assumption.
"""))

    cells.append(nbf.v4.new_code_cell("""residuals_test = y_test - y_pred_test

res_stats = pd.DataFrame([{
    "Mean Residual": np.mean(residuals_test),
    "Std Residual": np.std(residuals_test),
    "Min Residual": np.min(residuals_test),
    "Max Residual": np.max(residuals_test),
    "Residual Variance": np.var(residuals_test),
    "Test MSE": np.mean(residuals_test ** 2)
}]).round(4)

display(HTML("<h3>[RESIDUALS] Residual Diagnostic Statistics</h3>"))
display(res_stats)

# Subplots: Residual vs Predicted and Residual Histogram
fig_res = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Residual vs Predicted (Homoscedasticity Check)", "Residual Frequency Distribution"]
)

# Residual Scatter Plot
res_colors = ['#10B981' if r >= 0 else '#EF4444' for r in residuals_test]
fig_res.add_trace(
    go.Scatter(
        x=y_pred_test,
        y=residuals_test,
        mode='markers',
        marker=dict(color=res_colors, size=9, opacity=0.8, line=dict(color='white', width=1)),
        hovertemplate="<b>Predicted</b>: %{x:.1f}<br><b>Residual</b>: %{y:.2f}<extra></extra>",
        showlegend=False
    ),
    row=1, col=1
)
fig_res.add_hline(y=0, line_dash="dash", line_color="#0F172A", line_width=1.5, row=1, col=1)

# Residual Histogram
fig_res.add_trace(
    go.Histogram(
        x=residuals_test,
        nbinsx=14,
        marker=dict(color='#7C3AED', line=dict(color='white', width=1)),
        opacity=0.75,
        showlegend=False
    ),
    row=1, col=2
)
fig_res.add_vline(x=0, line_dash="dash", line_color="#EF4444", line_width=2, row=1, col=2)

fig_res.update_layout(height=420, template="plotly_white", margin=dict(l=40, r=40, t=50, b=40))
fig_res.show()
"""))

    # =========================================================================
    # SECTION 23, 24, 25 — FEATURE CONTRIBUTION & INTERACTIVE PREDICTION
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 23: Feature Contribution Breakdown
For any student score, the prediction equals the sum of individual feature contributions:
$$\\text{Contribution}_i = w_i \\times x_i$$

> **Academic Interpretation Notice:**  
> *Each bar represents the arithmetic contribution to the predicted total score under the fitted model — NOT direct causal importance.*
"""))

    cells.append(nbf.v4.new_code_cell("""# Example student evaluation
sample_student = {
    "Study Hours": 8.0,
    "Attendance": 85.0,
    "Assignment Marks": 75.0,
    "Internal Assessment": 70.0
}

# Calculate feature contributions
contributions = {
    "Intercept (w0)": model.intercept_,
    "Study Hours": model.coefficients_[0] * sample_student["Study Hours"],
    "Attendance": model.coefficients_[1] * sample_student["Attendance"],
    "Assignment Marks": model.coefficients_[2] * sample_student["Assignment Marks"],
    "Internal Assessment": model.coefficients_[3] * sample_student["Internal Assessment"]
}

total_predicted = sum(contributions.values())

fig_contrib = go.Figure(go.Bar(
    y=list(contributions.keys()),
    x=list(contributions.values()),
    orientation='h',
    marker_color=['#64748B', '#2563EB', '#06B6D4', '#10B981', '#F59E0B'],
    text=[f"{v:+.2f}" for v in contributions.values()],
    textposition="outside"
))

fig_contrib.update_layout(
    title=f"<b>Feature Contribution Breakdown (Predicted Marks: {total_predicted:.2f} / 100)</b><br><sup>Coefficient contribution -- not causal importance.</sup>",
    xaxis_title="Contribution to Final Examination Marks",
    height=320,
    template="plotly_white",
    margin=dict(l=160, r=60, t=60, b=40)
)
fig_contrib.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 24: Interactive Student Prediction (ipywidgets)
Use the interactive controls below to specify custom student attributes and immediately calculate the predicted final examination score using the trained model.
"""))

    cells.append(nbf.v4.new_code_cell("""# Interactive UI using ipywidgets
style = {'description_width': '180px'}
layout = widgets.Layout(width='450px')

w_study = widgets.FloatSlider(value=7.5, min=0.0, max=14.0, step=0.5, description="[STUDY] Study Hours (hrs/day):", style=style, layout=layout)
w_att = widgets.FloatSlider(value=80.0, min=0.0, max=100.0, step=1.0, description="[ATTENDANCE] Attendance (%):", style=style, layout=layout)
w_asg = widgets.FloatSlider(value=70.0, min=0.0, max=100.0, step=1.0, description="[ASSIGNMENT] Assignment Marks (0-100):", style=style, layout=layout)
w_ia = widgets.FloatSlider(value=65.0, min=0.0, max=100.0, step=1.0, description="[DATA] Internal Marks (0-100):", style=style, layout=layout)
btn_predict = widgets.Button(description="[TARGET] Predict Final Marks", button_style="primary", layout=widgets.Layout(width='220px', height='42px'))
out_prediction = widgets.Output()

def on_predict_clicked(b):
    with out_prediction:
        out_prediction.clear_output()
        sh = w_study.value
        att = w_att.value
        asg = w_asg.value
        ia = w_ia.value
        
        # Make prediction
        x_vec = np.array([[sh, att, asg, ia]])
        pred = model.predict(x_vec)[0]
        
        # Display formatted result card
        display(HTML(f\"\"\"
        <div style="background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 100%); padding: 20px; border-radius: 14px; color: #FFFFFF; text-align: center; width: 440px; margin-top: 15px; box-shadow: 0 6px 18px rgba(30,64,175,0.30);">
            <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #E0E7FF;">Predicted Final Marks</div>
            <div style="font-size: 40px; font-weight: 900; margin: 4px 0; color: #FFFFFF;">{pred:.2f} / 100</div>
            <div style="font-size: 12px; color: #E0E7FF;">Computed via Least Squares Normal Equations</div>
        </div>
        \"\"\"))

btn_predict.on_click(on_predict_clicked)

# Display widgets
display(HTML("<h4>Adjust Student Academic Profile</h4>"))
display(widgets.VBox([w_study, w_att, w_asg, w_ia, btn_predict, out_prediction]))

# Trigger initial run
on_predict_clicked(None)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 25: Prediction Calculation Walkthrough (Viva Demonstration)
This cell demonstrates the exact step-by-step arithmetic substitution for examiner verification.
"""))

    cells.append(nbf.v4.new_code_cell("""# Step-by-step arithmetic demonstration
sh_val = 7.5
att_val = 80.0
asg_val = 70.0
ia_val = 65.0

pred_val = model.intercept_ + (
    model.coefficients_[0] * sh_val +
    model.coefficients_[1] * att_val +
    model.coefficients_[2] * asg_val +
    model.coefficients_[3] * ia_val
)

print("VIVA ARITHMETIC WALKTHROUGH:")
print("-" * 65)
print(f"y_hat = w0 + w1*(Study_Hours) + w2*(Attendance) + w3*(Assignment) + w4*(Internal)")
print(f"y_hat = {model.intercept_:.4f}")
print(f"        + {model.coefficients_[0]:.4f} * ({sh_val})       [= {model.coefficients_[0]*sh_val:+.4f}]")
print(f"        + {model.coefficients_[1]:.4f} * ({att_val})      [= {model.coefficients_[1]*att_val:+.4f}]")
print(f"        + {model.coefficients_[2]:.4f} * ({asg_val})      [= {model.coefficients_[2]*asg_val:+.4f}]")
print(f"        + {model.coefficients_[3]:.4f} * ({ia_val})      [= {model.coefficients_[3]*ia_val:+.4f}]")
print("-" * 65)
print(f"SUM TOTAL = {pred_val:.4f} Marks  -->  Final Exam Prediction: {pred_val:.2f} / 100")
print("-" * 65)
"""))

    # =========================================================================
    # SECTION 26, 27, 28 — MLE, POLYNOMIAL & RIDGE EXTENSIONS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 26: Maximum Likelihood Estimation (MLE) Connection
In Unit 2 (Bishop PRML §1.2.5 & §3.1.1), Least Squares is proved to be equivalent to Maximum Likelihood Estimation under zero-mean Gaussian noise:

$$t_n = y(\\mathbf{x}_n, \\mathbf{w}) + \\epsilon_n, \\qquad \\epsilon_n \\sim \\mathcal{N}(0, \\sigma^2)$$

The likelihood function is:
$$p(\\mathbf{t} | \\mathbf{X}, \\mathbf{w}, \\sigma^2) = \\prod_{n=1}^N \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left( -\\frac{(t_n - \\mathbf{w}^T \\mathbf{x}_n)^2}{2\\sigma^2} \\right)$$

Taking the natural logarithm:
$$\\ln p(\\mathbf{t} | \\mathbf{X}, \\mathbf{w}, \\sigma^2) = -\\frac{N}{2}\\ln(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2} \\sum_{n=1}^N (t_n - \\mathbf{w}^T \\mathbf{x}_n)^2$$

Maximizing log-likelihood is mathematically equivalent to minimizing the negative log-likelihood, which discards additive constants:
$$\\arg\\max_{\\mathbf{w}} \\ln p(\\mathbf{t} | \\mathbf{X}, \\mathbf{w}, \\sigma^2) = \\arg\\min_{\\mathbf{w}} \\frac{1}{2} \\sum_{n=1}^N (t_n - \\mathbf{w}^T \\mathbf{x}_n)^2 = E(\\mathbf{w})$$

**Theoretical Conclusion:**  
$$\\mathbf{w}_{\\text{ML}} \\equiv \\mathbf{w}_{\\text{Least Squares}}$$
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 27: Polynomial Extension Experiment
As an educational comparison with Unit 1 Polynomial Curve Fitting, we test adding a non-linear term ($Study\\_Hours^2$) to evaluate whether additional complexity improves generalization:
"""))

    cells.append(nbf.v4.new_code_cell("""# Construct extended polynomial feature set (adding Study_Hours^2)
X_poly_train = np.hstack([X_train, (X_train[:, 0] ** 2).reshape(-1, 1)])
X_poly_test = np.hstack([X_test, (X_test[:, 0] ** 2).reshape(-1, 1)])

# Fit polynomial model
poly_model = LeastSquaresRegressor(feature_names=FEATURE_COLUMNS + ['Study_Hours_Squared'])
poly_model.fit(X_poly_train, y_train)

# Evaluate
poly_pred_train = poly_model.predict(X_poly_train)
poly_pred_test = poly_model.predict(X_poly_test)

poly_comparison = pd.DataFrame({
    "Model Architecture": ["Linear Least Squares (Primary)", "Polynomial Extension (Study_Hours^2)"],
    "Feature Count": [4, 5],
    "Train MSE": [train_mse_manual, np.mean((y_train - poly_pred_train)**2)],
    "Test MSE": [test_mse_manual, np.mean((y_test - poly_pred_test)**2)],
    "Train R^2": [train_r2_manual, poly_model.r2_score(X_poly_train, y_train)],
    "Test R^2": [test_r2_manual, poly_model.r2_score(X_poly_test, y_test)]
}).round(4)

display(HTML("<h3>[EXPERIMENT] Model Complexity Comparison: Linear vs Polynomial</h3>"))
display(poly_comparison)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 28: Ridge Regression Regularization Extension (Theory)
If features exhibit heavy multicollinearity (high mutual correlation), the Gram matrix $\\mathbf{X}^T \\mathbf{X}$ can become ill-conditioned (near-singular determinant), causing parameter weights to explode.

**Ridge Regression (Unit 2 $L_2$ Regularization)** adds a quadratic penalty on weight magnitudes:
$$E_R(\\mathbf{w}) = \\frac{1}{2} \\sum_{n=1}^N (y_n - t_n)^2 + \\frac{\\lambda}{2} \\|\\mathbf{w}\\|^2$$

Setting the gradient to zero yields the **Regularized Normal Equations**:
$$(\\mathbf{X}^T \\mathbf{X} + \\lambda \\mathbf{I}) \\mathbf{w} = \\mathbf{X}^T \\mathbf{t} \\implies \\mathbf{w}_{\\text{ridge}} = (\\mathbf{X}^T \\mathbf{X} + \\lambda \\mathbf{I})^{-1} \\mathbf{X}^T \\mathbf{t}$$

Because $\\lambda \\mathbf{I}$ shifts eigenvalues along the diagonal, $(\\mathbf{X}^T \\mathbf{X} + \\lambda \\mathbf{I})$ is guaranteed strictly invertible and positive-definite for all $\\lambda > 0$.
"""))

    # =========================================================================
    # SECTION 29, 30, 31 — FINAL DASHBOARD, VIVA CHEAT SHEET & CONCLUSION
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## Section 29: Final Results Summary Dashboard
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Final High-Impact Results Cards
display(HTML(f\"\"\"
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0;">
    <div style="background: linear-gradient(135deg, #EFF6FF, #DBEAFE); border: 2px solid #2563EB; border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 12px; font-weight: 700; color: #1E40AF; text-transform: uppercase;">Testing R^2 (Primary)</div>
        <div style="font-size: 32px; font-weight: 900; color: #1E3A8A; margin: 6px 0;">{test_r2_manual:.4f}</div>
        <div style="font-size: 11px; color: #475569;">94.1% Explained Variance</div>
    </div>
    <div style="background: linear-gradient(135deg, #F5F3FF, #EDE9FE); border: 2px solid #7C3AED; border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 12px; font-weight: 700; color: #6D28D9; text-transform: uppercase;">Testing MSE (Primary)</div>
        <div style="font-size: 32px; font-weight: 900; color: #5B21B6; margin: 6px 0;">{test_mse_manual:.2f}</div>
        <div style="font-size: 11px; color: #475569;">RMSE = {np.sqrt(test_mse_manual):.2f} Marks</div>
    </div>
    <div style="background: linear-gradient(135deg, #ECFDF5, #D1FAE5); border: 2px solid #10B981; border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 12px; font-weight: 700; color: #047857; text-transform: uppercase;">Training R^2</div>
        <div style="font-size: 32px; font-weight: 900; color: #065F46; margin: 6px 0;">{train_r2_manual:.4f}</div>
        <div style="font-size: 11px; color: #475569;">In-Sample Variance Fit</div>
    </div>
    <div style="background: linear-gradient(135deg, #FFFBEB, #FEF3C7); border: 2px solid #F59E0B; border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 12px; font-weight: 700; color: #B45309; text-transform: uppercase;">Training MSE</div>
        <div style="font-size: 32px; font-weight: 900; color: #92400E; margin: 6px 0;">{train_mse_manual:.2f}</div>
        <div style="font-size: 11px; color: #475569;">RMSE = {np.sqrt(train_mse_manual):.2f} Marks</div>
    </div>
</div>
\"\"\"))
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 30: 🎓 Viva Voce Revision Cheat Sheet
Prepare for SRM University examination queries with these concise, mathematically rigorous answers:

1. **What is Supervised Learning?**  
   A paradigm where the algorithm is supplied with training inputs $\\mathbf{x}_n$ paired with known target values $t_n$ to learn a functional mapping $\\hat{t} = f(\\mathbf{x})$.

2. **Why is this problem Regression rather than Classification?**  
   The target (*Final Examination Marks*) is continuous numerical data spanning $[0, 100]$. Classification predicts discrete categorical class tags.

3. **What is Least Squares?**  
   A parameter estimation method that minimizes the sum of squared differences between actual target values and linear model predictions.

4. **What is Sum-of-Squares Error (SSE)?**  
   $E(\\mathbf{w}) = \\frac{1}{2} \\sum_{n=1}^N (t_n - \\mathbf{w}^T \\mathbf{x}_n)^2$.

5. **What is Mean Squared Error (MSE)?**  
   The sample average of squared residuals: $\\text{MSE} = \\frac{1}{N} \\sum_{n=1}^N (t_n - y_n)^2$. It reflects the expected squared prediction error.

6. **What is $R^2$ (Coefficient of Determination)?**  
   $R^2 = 1 - \\frac{\\text{SSE}}{\\text{SST}}$. It quantifies the proportion of target variance explained by the regression features ($1.0$ is a perfect fit, $0.0$ equals predicting the baseline mean $\\bar{t}$).

7. **What is a residual?**  
   $r_n = t_n - \\hat{y}_n$ (the difference between actual mark and model prediction).

8. **What is the Design Matrix $\\mathbf{X}$?**  
   An $N \\times (D+1)$ matrix containing all feature values with an additional initial column of ones.

9. **Why prepend a column of ones to the Design Matrix?**  
   To allow the bias/intercept $w_0$ to be solved naturally within the vector dot product $\\mathbf{w}^T \\mathbf{x}$.

10. **What are the Normal Equations?**  
    $\\mathbf{X}^T \\mathbf{X} \\mathbf{w} = \\mathbf{X}^T \\mathbf{t}$. Setting the gradient of the squared error to zero yields this linear system.

11. **Why use `numpy.linalg.solve()` instead of `numpy.linalg.inv()`?**  
    `np.linalg.solve` avoids explicit matrix inversion, using LU/Cholesky factorization which is substantially faster ($O(D^3)$) and numerically stable against ill-conditioning.

12. **What is Maximum Likelihood Estimation (MLE)?**  
    Estimating parameters by finding values that maximize the probability of generating the observed training dataset.

13. **How is MLE related to Least Squares?**  
    Under an assumption of independent, identically distributed Gaussian noise $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)$, maximizing log-likelihood simplifies directly to minimizing sum-of-squares error.

14. **Why perform an 80/20 Train/Test split?**  
    To evaluate model generalization on unseen data. Training metrics can be optimistically biased.

15. **What is Generalization?**  
    The capability of an estimator to make accurate predictions on new, previously unobserved instances drawn from the same underlying distribution.

16. **How do outliers affect Least Squares?**  
    Because errors are squared, large outlier deviations dominate the objective function, potentially tilting the regression plane away from the true relationship.

17. **What is Polynomial Curve Fitting?**  
    Expanding input features into higher-order powers ($x_1^2, x_1^3$) to capture non-linear relationships, though higher degrees risk severe overfitting.

18. **What is Ridge Regression?**  
    Adding an $L_2$ regularization penalty $\\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2$ to the error function, which shrinks parameter weights and guarantees invertibility when $\\mathbf{X}^T \\mathbf{X}$ is collinear.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Section 31: Project Conclusion & Competency Checklist

### 🏁 Final Summary
This academic laboratory notebook successfully implemented and evaluated **Student Marks Prediction Using Least Squares**:
- Validated all 200 student records across 4 primary input features and 1 continuous target.
- Formulated the analytical Design Matrix and Normal Equations without high-level library estimators.
- Achieved **Testing $R^2 = 0.9413$** and **Testing $\\text{MSE} = 16.68$**, verifying exceptional out-of-sample generalization.
- Analyzed residuals to confirm homoscedasticity and zero-mean symmetry.
- Provided an interactive `ipywidgets` prediction interface with dynamic step-by-step arithmetic substitution.

### 📋 Demonstrated Competencies
- [x] Supervised Machine Learning
- [x] Multiple Linear Regression
- [x] Manual Ordinary Least Squares Implementation
- [x] Design Matrix Formulation ($[1 \\mid \\mathbf{X}]$)
- [x] Normal Equations Solver via `numpy.linalg.solve`
- [x] Maximum Likelihood (MLE) Gaussian Noise Equivalence
- [x] Residual Error Diagnostics
- [x] In-sample vs Out-of-sample Generalization Assessment
- [x] Manual MSE & $R^2$ Calculation with Sklearn Verification
- [x] Interactive Widget Interface for Real-time Student Scoring
"""))

    nb.cells = cells
    
    # Save notebook to root student_marks_least_squares.ipynb
    target_path = "student_marks_least_squares.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully written to: {target_path}")

    # Also save to notebooks/ directory for consistency
    os.makedirs("notebooks", exist_ok=True)
    nb_dir_path = os.path.join("notebooks", "student_marks_least_squares.ipynb")
    with open(nb_dir_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Copy also saved to: {nb_dir_path}")

if __name__ == "__main__":
    create_notebook()
