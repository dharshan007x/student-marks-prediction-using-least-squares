# Student Marks Prediction Using Least Squares

**Predicting Final Examination Marks Using Multiple Linear Least Squares Regression**

---

## Project Objective

Develop a Student Marks Prediction system that predicts a student's **Final Examination Marks** using:

- **Study Hours** — average daily study time
- **Attendance** — class attendance percentage
- **Assignment Marks** — marks obtained in assignments
- **Internal Assessment Marks** — internal assessment scores

The Least Squares regression model is implemented **from scratch** using the Normal Equations — no `sklearn.LinearRegression()`.

## Problem Statement

> Predict students' final examination marks using study hours, attendance, assignment marks, and internal assessment marks. Implement the Least Squares method from scratch to estimate regression coefficients and calculate MSE and R².

## Academic Concepts Demonstrated

### Unit 1

- Machine Learning & Supervised Learning
- Regression & Linear Models
- Training / Test Data & Generalization
- Features & Target Variables
- Sum-of-Squares Error
- Polynomial Curve Fitting

### Unit 2

- Maximum Likelihood Estimation
- Least Squares & Normal Equations
- Regression Coefficients & Residuals
- Model Evaluation (MSE, R²)
- Ridge Regression (extension discussion)
- Bayesian Linear Regression (extension discussion)

## Dataset

**Type:** Synthetic / project dataset created for demonstrating Least Squares regression.

| Column | Description | Range |
|--------|-------------|-------|
| `Study_Hours` | Daily study hours | 0 – 24 |
| `Attendance` | Attendance % | 0 – 100 |
| `Assignment_Marks` | Assignment marks | 0 – 100 |
| `Internal_Assessment_Marks` | Internal assessment marks | 0 – 100 |
| `Final_Exam_Marks` | **Target** — final exam marks | 0 – 100 |

- **200 student records** (160 training / 40 testing)

> **Note:** This dataset is synthetic. It is NOT collected from real students.

## Mathematical Formulation

### Linear Model

```
y(x, w) = w₀ + w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄
```

### Design Matrix

```
X = [1  x₁₁  x₂₁  x₃₁  x₄₁]
    [1  x₁₂  x₂₂  x₃₂  x₄₂]
    [⋮   ⋮    ⋮    ⋮    ⋮  ]
    [1  x₁N  x₂N  x₃N  x₄N]
```

### Sum-of-Squares Error

```
E(w) = ½ Σₙ [y(xₙ, w) − tₙ]²
```

### Normal Equations

```
XᵀXw = Xᵀt  →  w = (XᵀX)⁻¹Xᵀt
```

Solved using `numpy.linalg.solve()` for numerical stability.

### Evaluation Metrics

- **MSE** = (1/N) Σ(tₙ − yₙ)²
- **R²** = 1 − [Σ(tₙ − yₙ)² / Σ(tₙ − t̄)²]

## Project Architecture

```
student-marks-least-squares/
│
├── data/
│   └── student_marks.csv           # Dataset (200 records)
│
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py          # Synthetic data generator
│   ├── data_loader.py               # Loading, validation, splitting
│   ├── least_squares.py             # LeastSquaresRegressor class
│   ├── evaluation.py                # MSE, R², MAE, RMSE, residuals
│   ├── visualizations.py            # Matplotlib plotting utilities
│   ├── utils.py                     # Formatting and helper functions
│   └── main.py                      # Command-line program
│
├── app/
│   └── streamlit_app.py             # Streamlit dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py                # Model fitting and shape tests
│   ├── test_predictions.py          # Prediction and metric tests
│   └── test_math.py                 # Mathematical correctness tests
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── feature_relationships.png
│   └── model_statistics.csv
│
├── notebooks/
│   └── student_marks_analysis.ipynb  # Step-by-step Jupyter notebook
│
├── report/
│   ├── project_report.md            # Full project report
│   └── viva_questions.md            # 38 viva Q&A
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
# Clone or download the project
cd student-marks-least-squares

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Command-Line Program

```bash
python src/main.py
```

This displays the complete pipeline: dataset loading, validation, train/test split, design matrix, Normal Equations, coefficients, predictions, evaluation, and residual analysis.

### Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Navigate through pages: Home, Dataset, Least Squares Mathematics, Model Training, Prediction, Evaluation, Residual Analysis, Analysis, About.

### Run Tests

```bash
pytest tests/ -v
```

39 tests covering model fitting, predictions, evaluation metrics, design matrix, normal equations, and mathematical correctness.

## Key Results

All numerical results are computed live from the dataset — nothing is hardcoded.

Run `python src/main.py` to see:

- Learned regression coefficients
- Model equation
- Training MSE, R²
- **Testing MSE, R²** (primary result)
- Residual analysis
- Prediction for a sample student

## Screenshots

*Run the Streamlit app to see the interactive dashboard.*

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Core | NumPy, Pandas |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Testing | Pytest |
| Data Splitting | scikit-learn (train_test_split only) |

> **Important:** The Least Squares regression model is implemented **manually from scratch**. scikit-learn is used only for train/test splitting.

## Limitations

1. Linearity assumption — may miss nonlinear patterns
2. Sensitivity to outliers — squared errors amplify extreme values
3. Synthetic data — results may not transfer to real student populations
4. No feature interactions modeled
5. Fixed set of four features

## Future Scope

1. Real student data collection and validation
2. Ridge and Lasso regression implementation
3. Feature interaction and polynomial terms
4. Cross-validation for robust evaluation
5. Web API deployment
6. Additional features (lab marks, participation, etc.)

---

*This project is developed for the Machine Learning course, demonstrating Least Squares regression with full mathematical transparency.*
