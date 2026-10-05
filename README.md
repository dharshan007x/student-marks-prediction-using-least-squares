# Student Marks Prediction Using Least Squares

**Predicting Final Examination Marks Using Multiple Linear Least Squares Regression**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dharshan007x/student-marks-prediction-using-least-squares/blob/main/student_marks_least_squares.ipynb)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎓 Primary Deliverable: Google Colab Laboratory Notebook

The primary deliverable for this SRM University Machine Learning course project is the standalone, presentation-ready Google Colab notebook:

👉 **[`student_marks_least_squares.ipynb`](https://colab.research.google.com/github/dharshan007x/student-marks-prediction-using-least-squares/blob/main/student_marks_least_squares.ipynb)**

### How to Run in Google Colab:
1. Click the **[Open In Colab](https://colab.research.google.com/github/dharshan007x/student-marks-prediction-using-least-squares/blob/main/student_marks_least_squares.ipynb)** badge above.
2. In Colab, click **Runtime** &rarr; **Run all** (or press `Ctrl + F9`).
3. The multi-tier resilient data loader will automatically fetch `student_marks.csv` directly from this GitHub repository.
4. Interact with the live `ipywidgets` sliders in Section 25 to predict final exam marks in real time.

---

## Project Objective

Develop a mathematically transparent Student Marks Prediction system that predicts a student's **Final Examination Marks** using four core continuous academic features:

1. **Study Hours ($x_1$)** — Average daily self-study time (0 to 14 hrs/day)
2. **Attendance ($x_2$)** — Lecture attendance percentage (0% to 100%)
3. **Assignment Marks ($x_3$)** — Continuous coursework assignment score (0 to 100)
4. **Internal Assessment Marks ($x_4$)** — Periodic mid-term examination score (0 to 100)

**Target Variable ($t$):** **Final Examination Marks** (0 to 100)

The Least Squares regression model is implemented **completely from scratch using NumPy** via the analytical Normal Equations:
$$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t} \implies \mathbf{w} = \text{solve}(\mathbf{X}^T \mathbf{X}, \mathbf{X}^T \mathbf{t})$$

Strictly no Scikit-Learn estimator classes (`LinearRegression`) or deep learning frameworks (TensorFlow, PyTorch) are used for model training.

---

## Academic Concepts Demonstrated (SRM ML Curriculum)

| Unit | Core Topic | Demonstration in Project |
| :--- | :--- | :--- |
| **Unit 1** | **Machine Learning Paradigm** | Learning mapping from 4 continuous inputs to continuous final exam marks. |
| **Unit 1** | **Supervised Learning** | Training dataset provides labeled feature vectors $\mathbf{x}_n$ paired with ground-truth marks $t_n$. |
| **Unit 1** | **Regression vs Classification** | Continuous target domain $[0, 100]$; metric Euclidean distance optimization rather than cross-entropy. |
| **Unit 1** | **Train / Test Split & Generalization** | Strict 80/20 data split ($N_{\text{train}}=160$, $N_{\text{test}}=40$) ensuring unbiased out-of-sample evaluation. |
| **Unit 1** | **Sum-of-Squares Error (SSE)** | $E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (t_n - \mathbf{w}^T \mathbf{x}_n)^2$ convex loss function. |
| **Unit 1** | **Polynomial Curve Fitting** | Higher-order feature comparison ($\text{Study\_Hours}^2$) demonstrating model complexity tradeoffs. |
| **Unit 2** | **Linear Models for Regression** | Linear combination $\hat{t}(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4$. |
| **Unit 2** | **Maximum Likelihood (MLE)** | Formal proof that minimizing squared error is identical to maximizing likelihood under Gaussian noise. |
| **Unit 2** | **Normal Equations Solver** | Analytical gradient $\nabla E(\mathbf{w}) = 0 \implies \mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$ solved via `np.linalg.solve()`. |
| **Unit 2** | **Regression Coefficients** | Parameter interpretation: intercept $w_0$ and feature slope weights $w_1, w_2, w_3, w_4$. |
| **Unit 2** | **Residual Diagnostics** | $r_n = t_n - \hat{y}_n$; verifying mean $\approx 0$, constant variance, and absence of heteroscedasticity. |
| **Unit 2** | **Model Evaluation (MSE & $R^2$)** | Manual MSE and $R^2$ verified against Scikit-Learn to 6 decimal places. |
| **Unit 2** | **Regularization (Ridge)** | Mathematical discussion of $L_2$ penalty $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1}$ to guard against multicollinearity. |

---

## Dataset Overview

- **Type:** Synthetic academic dataset generated specifically for course demonstration.
- **Records:** 200 student samples (160 training / 40 testing).
- **Integrity:** Zero missing values, zero duplicates, all values within valid physical bounds.

| Column | Description | Range | Role |
| :--- | :--- | :--- | :--- |
| `Study_Hours` | Average daily study time | 0.5 – 14.0 hrs | Feature ($x_1$) |
| `Attendance` | Lecture attendance percentage | 35.0% – 100.0% | Feature ($x_2$) |
| `Assignment_Marks` | Continuous assessment marks | 15.0 – 100.0 | Feature ($x_3$) |
| `Internal_Assessment_Marks` | Periodic mid-term internal marks | 20.0 – 100.0 | Feature ($x_4$) |
| `Final_Exam_Marks` | End-semester examination score | 0.0 – 100.0 | **Target ($t$)** |

> **Academic Notice:** *This dataset is synthetic and does not represent private or identifiable student records.*

---

## Mathematical Formulation

### 1. Design Matrix Formulation
To estimate the intercept $w_0$ alongside slope coefficients, we prepend a column of ones:
$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{12} & x_{13} & x_{14} \\ 1 & x_{21} & x_{22} & x_{23} & x_{24} \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ 1 & x_{N1} & x_{N2} & x_{N3} & x_{N4} \end{bmatrix} \in \mathbb{R}^{N \times 5}$$

### 2. Normal Equations
Setting the gradient of Sum-of-Squares Error $\nabla_{\mathbf{w}} E(\mathbf{w}) = \mathbf{0}$ yields:
$$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$$

### 3. Numerical Solver
Rather than computing the explicit matrix inverse $(\mathbf{X}^T \mathbf{X})^{-1}$ (which is computationally expensive and prone to numerical instability), we solve the system directly using LU/Cholesky factorization:
```python
w = np.linalg.solve(X_design.T @ X_design, X_design.T @ y_train)
```

---

## Key Results & Verification

All metrics are computed live from the dataset:

| Metric | Manual Implementation | Scikit-Learn Verification | Difference | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Training MSE** | `16.9806` | `16.9806` | `0.000000` | **Exact Match [PASS]** |
| **Testing MSE (Primary)** | `16.6786` | `16.6786` | `0.000000` | **Exact Match [PASS]** |
| **Training $R^2$** | `0.9222` | `0.9222` | `0.000000` | **Exact Match [PASS]** |
| **Testing $R^2$ (Primary)** | `0.9413` | `0.9413` | `0.000000` | **Exact Match [PASS]** |

### Learned Regression Equation:
$$\hat{y} = 11.4137 + 3.4112 \cdot (\text{Study\_Hours}) + 0.1040 \cdot (\text{Attendance}) + 0.2792 \cdot (\text{Assignment}) + 0.3168 \cdot (\text{Internal})$$

---

## 31 Structured Notebook Sections

The deliverable notebook [`student_marks_least_squares.ipynb`](./student_marks_least_squares.ipynb) contains all 31 comprehensive sections:

1. **Project Title & Academic Banner**
2. **Academic Curriculum Concept Mapping** (Units 1 & 2)
3. **Import Libraries & Setup Environment**
4. **Dataset Overview & Academic Integrity Notice**
5. **Multi-Tier Resilient Data Loader** (Local / GitHub / Synthetic Fallback)
6. **Data Integrity & Quality Validation** (Zero nulls, bounds checks)
7. **Exploratory Data Analysis — Interactive Visualizations**
8. **Summary Statistics & Distribution Analysis**
9. **Correlation Analysis & Heatmap**
10. **Feature Matrix ($\mathbf{X}$) and Target Vector ($\mathbf{t}$) Definition**
11. **Mathematical Model Formulation** ($\mathbf{w}^T \mathbf{x}$)
12. **Design Matrix Construction** ($[1 \mid \mathbf{X}]$)
13. **Least Squares Error Function** (Sum-of-Squares Error)
14. **Analytical Normal Equations Derivation** ($\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$)
15. **Manual `LeastSquaresRegressor` Implementation** (Pure NumPy)
16. **Train/Test Data Splitting** (80% Train, 20% Test)
17. **Model Training & Visible Matrix Operations** (Gram matrix, Moment vector)
18. **Learned Parameter Weights & Formatted Equation**
19. **Predictions on Unseen Test Dataset**
20. **Manual MSE Calculation & Step-by-Step Breakdown**
21. **Manual $R^2$ Score Calculation & SST/SSE Breakdown**
22. **Scikit-Learn Verification & Exact Match Validation Table**
23. **Actual vs Predicted Marks Visualizations** (Plotly + Matplotlib)
24. **Residual Diagnostics & Assumption Checks** (Zero-mean, homoscedasticity)
25. **Interactive Prediction Interface** (`ipywidgets` Sliders)
26. **Feature Contribution Breakdown for Individual Students**
27. **Higher-Order Extension: Polynomial Curve Fitting Comparison**
28. **Regularization Extension: Ridge Regression Discussion**
29. **Bayesian Linear Regression Educational Overview**
30. **Viva Voce Revision Cheat Sheet** (18 rigorous Q&As)
31. **Project Conclusion & Competency Checklist**

---

## Project Structure

```
student-marks-prediction-using-least-squares/
│
├── student_marks_least_squares.ipynb  # PRIMARY DELIVERABLE (Pre-rendered Google Colab Notebook)
├── build_colab_notebook.py           # Automated notebook generator script
│
├── data/
│   └── student_marks.csv              # Synthetic dataset (200 records)
│
├── notebooks/
│   ├── student_marks_least_squares.ipynb  # Synced Colab notebook
│   └── student_marks_analysis.ipynb       # Supplementary Jupyter analysis
│
├── src/
│   ├── least_squares.py               # Standalone LeastSquaresRegressor class
│   ├── data_loader.py                 # Data loader, validation, and splitting
│   ├── evaluation.py                  # Evaluation metrics (MSE, RMSE, MAE, R²)
│   ├── generate_dataset.py            # Dataset generator script
│   ├── main.py                        # Complete CLI demonstration
│   ├── utils.py                       # Formatting helpers
│   └── visualizations.py              # Publication-ready plotting utilities
│
├── tests/
│   ├── test_math.py                   # Normal equations & design matrix tests
│   ├── test_model.py                  # Model fit, predict, and shape tests
│   └── test_predictions.py            # Prediction accuracy & metric tests
│
├── report/
│   ├── project_report.md              # Academic project documentation
│   └── viva_questions.md              # 38 comprehensive viva Q&A
│
├── requirements.txt                   # Environment dependencies
└── README.md                          # Repository documentation & guide
```

---

## Quickstart Guide

### Option 1: Run in Google Colab (Recommended)
Click the badge below to run the complete notebook with interactive widgets in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dharshan007x/student-marks-prediction-using-least-squares/blob/main/student_marks_least_squares.ipynb)

### Option 2: Run Locally via Jupyter
```bash
# Clone the repository
git clone https://github.com/dharshan007x/student-marks-prediction-using-least-squares.git
cd student-marks-prediction-using-least-squares

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook student_marks_least_squares.ipynb
```

### Option 3: Run the CLI Pipeline
```bash
python src/main.py
```

### Option 4: Run the Test Suite
```bash
pytest tests/ -v
```
*(All 39 unit tests pass successfully).*

---

## Academic Assessment & Viva Preparation

A dedicated **Viva Voce Revision Cheat Sheet** is embedded in **Section 30** of the Colab notebook, providing clear, mathematically precise answers to core questions:
- *Why Regression instead of Classification?*
- *Why prepend a column of ones to the Design Matrix?*
- *Why solve $(\mathbf{X}^T \mathbf{X}) \mathbf{w} = \mathbf{X}^T \mathbf{t}$ with `np.linalg.solve` instead of matrix inversion `np.linalg.inv`?*
- *How does Maximum Likelihood Estimation (MLE) relate to Least Squares under Gaussian noise?*
- *What is the interpretation of $R^2 = 0.9413$?*
- *What are residuals and what does homoscedasticity mean?*
- *How do Ridge regression and Polynomial curve fitting extend Ordinary Least Squares?*

---

## License

This project is licensed under the MIT License for educational and academic use at SRM Institute of Science and Technology.
