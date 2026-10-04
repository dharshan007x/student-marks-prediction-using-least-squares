# Student Marks Prediction Using Least Squares — Project Report

## 1. Title

**Student Marks Prediction Using Least Squares**

*Predicting Final Examination Marks Using Multiple Linear Least Squares Regression*

---

## 2. Abstract

This project implements a **Student Marks Prediction** system using the **Least Squares regression** method. The system predicts a student's Final Examination Marks based on four input features: Study Hours, Attendance, Assignment Marks, and Internal Assessment Marks. The Least Squares regression model is implemented **from scratch** using the Normal Equations, without relying on pre-built machine learning libraries for the core algorithm. The project demonstrates key concepts from Unit 1 (Supervised Learning, Regression, Sum-of-Squares Error, Polynomial Curve Fitting) and Unit 2 (Maximum Likelihood Estimation, Least Squares, Normal Equations, Regression Coefficients, Model Evaluation) of the Machine Learning curriculum.

---

## 3. Problem Statement

**Predict students' final examination marks using study hours, attendance, assignment marks, and internal assessment marks. Implement the Least Squares method from scratch to estimate regression coefficients and calculate MSE and R².**

---

## 4. Project Objective

Develop a Student Marks Prediction system that:

1. Constructs the feature/design matrix
2. Estimates regression coefficients using the Normal Equations
3. Generates predicted final examination marks
4. Calculates prediction residuals
5. Evaluates the model using Mean Squared Error (MSE) and R² Score
6. Allows prediction for new students
7. Provides visual analysis of actual versus predicted marks
8. Clearly explains the mathematics behind every major step

---

## 5. Motivation

Predicting student performance is an important application of machine learning in education. By identifying the factors that influence examination marks, educators can:

- Identify at-risk students early
- Understand the relative importance of different academic inputs
- Design targeted interventions

From an academic perspective, this problem is well-suited for demonstrating the Least Squares regression method because:

- The target variable (Final Exam Marks) is continuous
- The relationship between inputs and the target is approximately linear
- The Normal Equations provide a closed-form solution
- The resulting model is easily interpretable

---

## 6. Machine Learning Background

### Unit 1 Concepts

**Machine Learning** is a field of study that gives computers the ability to learn from data without being explicitly programmed. In this project, the computer learns the relationship between student features and final marks from training data.

**Supervised Learning** uses labeled training examples (input-output pairs) to learn a mapping function. Our training data contains both input features and the corresponding final marks.

**Regression** is a type of supervised learning where the target variable is continuous. Since Final Examination Marks are real-valued numbers, this is a regression problem.

**Training Data** is used to learn the model parameters. **Test Data** is used to evaluate how well the model generalizes to unseen examples.

**Generalization** refers to the model's ability to perform well on new, unseen data — not just the training data it was fitted on.

**Features** (Input Variables) are the measurable properties used for prediction: Study Hours, Attendance, Assignment Marks, Internal Assessment Marks.

**Target** (Response Variable) is what we want to predict: Final Examination Marks.

**Sum-of-Squares Error** measures the total squared difference between predictions and actual values. This is the objective function that Least Squares minimizes.

**Polynomial Curve Fitting** (Unit 1) demonstrates that model coefficients can be estimated by minimizing a sum-of-squares error. Our project uses a linear model, but we optionally extend it with a polynomial term to demonstrate model complexity and overfitting.

### Unit 2 Concepts

**Maximum Likelihood Estimation** provides a probabilistic justification for the Least Squares method. Under Gaussian noise assumptions, maximizing the likelihood is equivalent to minimizing the sum-of-squares error.

**Least Squares** is the primary algorithm: find the weight vector **w** that minimizes the sum of squared prediction errors.

**Normal Equations** provide the closed-form solution: **w = (X^T X)^{-1} X^T t**

**Regression Coefficients** are the learned parameters that define the linear relationship between inputs and the target.

**Residuals** are the differences between actual and predicted values.

**Model Evaluation** uses MSE and R² to quantify prediction quality.

**Ridge Regression** and **Bayesian Linear Regression** are discussed as theoretical extensions that address limitations of plain Least Squares.

---

## 7. Dataset Description

### Source

This is a **synthetic / project dataset** created specifically for demonstrating the Least Squares regression method. It is NOT collected from real students and should NOT be presented as real-world collected student data.

### Structure

| Column | Description | Range |
|--------|-------------|-------|
| Study_Hours | Average daily study hours | 0 – 24 |
| Attendance | Attendance percentage | 0 – 100 |
| Assignment_Marks | Assignment marks | 0 – 100 |
| Internal_Assessment_Marks | Internal assessment marks | 0 – 100 |
| Final_Exam_Marks | Final examination marks (target) | 0 – 100 |

### Size

- **Total records**: 200 students
- **Training set**: 160 students (80%)
- **Testing set**: 40 students (20%)

---

## 8. Feature Description

| Feature | Symbol | Type | Description |
|---------|--------|------|-------------|
| Study Hours | x₁ | Continuous | Hours spent studying per day |
| Attendance | x₂ | Continuous | Class attendance percentage |
| Assignment Marks | x₃ | Continuous | Marks obtained in assignments |
| Internal Assessment Marks | x₄ | Continuous | Marks in internal assessments |

---

## 9. Target Variable

| Variable | Symbol | Type | Description |
|----------|--------|------|-------------|
| Final Examination Marks | t | Continuous | Target: final exam score (0-100) |

---

## 10. Data Preprocessing

### Steps Performed

1. **Dataset Loading**: CSV file loaded using Pandas
2. **Data Validation**: Checked for missing values, duplicates, invalid numerics, out-of-range values
3. **Feature Selection**: Selected the four required input features
4. **Target Selection**: Selected Final_Exam_Marks as the target
5. **Train/Test Split**: 80/20 split with random_state=42 for reproducibility

### Why No Scaling/Normalization?

The basic Least Squares formulation does not require feature scaling. The Normal Equations produce the optimal solution regardless of feature scale. Scaling would change the interpretation of the regression coefficients. However, scaling can improve numerical stability in practice — for this dataset size and feature range, it is not necessary.

---

## 11. Least Squares Theory

### Linear Regression Model

$$y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4$$

### Design Matrix

For N training samples:

$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{21} & x_{31} & x_{41} \\ 1 & x_{12} & x_{22} & x_{32} & x_{42} \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ 1 & x_{1N} & x_{2N} & x_{3N} & x_{4N} \end{bmatrix}$$

### Sum-of-Squares Error

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} \left[ y(\mathbf{x}_n, \mathbf{w}) - t_n \right]^2$$

### Normal Equations

Setting the gradient to zero:

$$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$$

Solved using `numpy.linalg.solve()` for numerical stability.

### Maximum Likelihood Connection

Under Gaussian noise:

$$t_n = y(\mathbf{x}_n, \mathbf{w}) + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma^2)$$

Maximizing the log-likelihood is equivalent to minimizing the sum-of-squares error, establishing that **Least Squares estimation = Maximum Likelihood estimation** under Gaussian noise assumptions.

---

## 12. Model Implementation

The `LeastSquaresRegressor` class implements:

- `fit(X, y)`: Trains the model using the Normal Equations
- `predict(X)`: Generates predictions
- `residuals(X, y)`: Computes residuals
- `sum_squared_error(X, y)`: Calculates SSE
- `mse(X, y)`: Mean Squared Error
- `r2_score(X, y)`: R² Score
- `get_model_equation()`: Returns the learned equation
- `get_design_matrix(X)`: Constructs the design matrix
- `get_normal_equation_components(X, y)`: Returns X, X^T, X^TX, X^Ty, w

---

## 13. Results

*Note: All values below are computed from the actual dataset. They are not fabricated.*

### Learned Coefficients

The regression coefficients are learned by the model during training and are displayed in the application output.

### Evaluation Metrics

| Metric | Training | Testing |
|--------|----------|---------|
| MSE | Computed from model | Computed from model |
| R² | Computed from model | Computed from model |

The actual numerical values are displayed when running `python src/main.py` or the Streamlit dashboard.

---

## 14. Limitations

1. **Linearity assumption**: The model assumes a linear relationship; nonlinear patterns may be missed
2. **Sensitivity to outliers**: Squared errors give disproportionate weight to outliers
3. **Feature independence**: Does not model interactions between features
4. **Synthetic data**: Results may not generalize to real student populations
5. **Fixed features**: Only four input features are considered

---

## 15. Polynomial Extension

An optional polynomial extension adds a squared term for Study Hours:

$$y = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 + w_5 x_1^2$$

This demonstrates the Unit 1 concept of polynomial curve fitting and the trade-off between model complexity and generalization.

---

## 16. Ridge Regression as an Extension

Ridge Regression adds L2 regularization:

$$E_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2} \sum (y_n - t_n)^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

This shrinks coefficients toward zero, reducing overfitting. It is NOT the main algorithm — it is discussed as a theoretical extension consistent with Unit 2 material.

---

## 17. Conclusion

This project successfully demonstrates the application of Least Squares regression to predict student final examination marks. The model is implemented from scratch using the Normal Equations, providing full mathematical transparency. Key ML concepts from both Unit 1 and Unit 2 are demonstrated through the implementation, evaluation, and analysis.

---

## 18. Future Scope

1. Use real student data for validation
2. Implement Ridge and Lasso regression
3. Add feature interaction terms
4. Explore cross-validation for more robust evaluation
5. Build a web API for deployment
6. Add more student features (e.g., participation, lab marks)

---

## 19. References

1. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer. — Chapters 1 and 3 (Linear Models for Regression)
2. Course Material — Unit 1: Introduction to Machine Learning, Supervised Learning, Polynomial Curve Fitting
3. Course Material — Unit 2: Maximum Likelihood, Least Squares, Normal Equations
4. NumPy Documentation — `numpy.linalg.solve()`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
