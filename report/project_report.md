# Student Marks Prediction Using Least Squares
## Comprehensive Academic Project Report

**Course:** Machine Learning (SRM Institute of Science and Technology)  
**Project Title:** STUDENT MARKS PREDICTION USING LEAST SQUARES  
**Subtitle:** Predicting Final Examination Marks Using Multiple Linear Least Squares Regression  
**Primary Deliverables:** 
- Google Colab Notebook: [`student_marks_least_squares.ipynb`](../student_marks_least_squares.ipynb)
- Source Implementation: `src/least_squares.py`
- Test Suite: `tests/` (39/39 Passed)

---

## 1. Executive Summary & Abstract

This report presents the complete design, mathematical derivation, software implementation, and empirical evaluation of an academic **Student Marks Prediction System** using the **Ordinary Least Squares (OLS)** regression method. The objective is to predict students' comprehensive **Final Examination Marks** ($t \in [0, 100]$) based on four continuous academic behavioral and continuous evaluation metrics:
1. **Study Hours** ($x_1$): Average daily self-directed study duration.
2. **Attendance** ($x_2$): Lecture attendance percentage.
3. **Assignment Marks** ($x_3$): Continuous coursework assessment score.
4. **Internal Assessment Marks** ($x_4$): Mid-semester periodic test score.

The core Machine Learning algorithm is formulated and implemented **from scratch using pure NumPy**, strictly relying on the closed-form **Normal Equations**:
$$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t} \implies \mathbf{w} = \text{solve}(\mathbf{X}^T \mathbf{X}, \mathbf{X}^T \mathbf{t})$$

The implementation completely eschews pre-packaged estimators (such as `sklearn.linear_model.LinearRegression`). Scikit-Learn is employed strictly for the unbiased 80/20 train/test partition and for 6-decimal-place numerical verification of the manual implementation.

On an independently evaluated test set ($N=40$), the model achieved a **Coefficient of Determination ($R^2$) of 0.9413** and a **Mean Squared Error (MSE) of 16.6786** (RMSE = 4.08 marks). These metrics match Scikit-Learn to $0.000000$ difference, proving 100% mathematical and numerical fidelity.

---

## 2. Problem Statement & Motivation

### 2.1 Problem Statement
> **Formally:** Given an academic dataset $\mathcal{D} = \{(\mathbf{x}_n, t_n)\}_{n=1}^N$ comprising $N=200$ student records, where $\mathbf{x}_n = [x_{n1}, x_{n2}, x_{n3}, x_{n4}]^T \in \mathbb{R}^4$ denotes the four-dimensional continuous feature vector and $t_n \in [0, 100]$ denotes the final exam mark, construct an optimal parameter vector $\mathbf{w}^* = [w_0, w_1, w_2, w_3, w_4]^T \in \mathbb{R}^5$ by minimizing the Sum-of-Squares Error function via the Normal Equations. Evaluate model generalization on an unseen test set using MSE and $R^2$.

### 2.2 Pedagogical Motivation
In higher education institutions such as SRM Institute of Science and Technology, academic performance depends on a multifaceted combination of student discipline (study hours and class attendance) and periodic formative evaluations (assignments and internal tests). Developing a predictive model serves two vital purposes:
1. **Early Academic Intervention:** Enables educators to identify students struggling across internal metrics well before final exams take place.
2. **Transparent ML Demonstration:** Unlike "black-box" deep learning or complex ensemble trees, multiple linear regression via Ordinary Least Squares provides a fully interpretable, mathematically transparent closed-form solution where every parameter weight has a direct physical and academic meaning.

---

## 3. Academic Curriculum Concept Mapping (SRM ML Units 1 & 2)

This project directly maps to the core concepts prescribed in the **SRM University Machine Learning Syllabus**:

| Unit | Syllabus Topic | Theoretical Concept | Concrete Demonstration in This Project |
| :--- | :--- | :--- | :--- |
| **Unit 1** | **Machine Learning Paradigm** | Inductive learning from experience | Learning the functional mapping $f: \mathbb{R}^4 \to \mathbb{R}$ from historical student profiles. |
| **Unit 1** | **Supervised Learning** | Labeled data pairs $(\mathbf{x}_n, t_n)$ | Training dataset provides both the 4 input features and ground-truth final marks. |
| **Unit 1** | **Regression vs Classification** | Continuous vs discrete target space | Target $t$ is continuous numeric marks $[0, 100]$; metric distance (MSE) optimization rather than cross-entropy loss. |
| **Unit 1** | **Train / Test Splitting** | Out-of-sample generalization | Strict 80/20 partition ($N_{\text{train}}=160, N_{\text{test}}=40$) with `random_state=42` to prevent data leakage. |
| **Unit 1** | **Sum-of-Squares Error (SSE)** | Objective loss function | $E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (t_n - \mathbf{w}^T \mathbf{x}_n)^2$ formulated as the convex optimization target. |
| **Unit 1** | **Polynomial Curve Fitting** | Non-linear feature expansion | Evaluated quadratic model extension ($x_1^2$) to analyze model complexity vs overfitting tradeoffs. |
| **Unit 2** | **Linear Models for Regression** | Linear parametric combinations | Affine hypothesis: $\hat{t}(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{j=1}^4 w_j x_j = \mathbf{w}^T \mathbf{x}$. |
| **Unit 2** | **Maximum Likelihood Estimation** | Probabilistic foundation of OLS | Proven: Under i.i.d. Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$, maximizing log-likelihood is mathematically identical to minimizing SSE. |
| **Unit 2** | **Normal Equations Derivation** | Analytical stationary point | $\nabla_{\mathbf{w}} E(\mathbf{w}) = \mathbf{0} \implies \mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$ derived from first principles. |
| **Unit 2** | **Numerical Linear Algebra** | Computational stability | Solved via LU/Cholesky factorization (`np.linalg.solve`) instead of explicit matrix inversion (`np.linalg.inv`). |
| **Unit 2** | **Residual Error Diagnostics** | Assumption validation | Analyzed residuals $r_n = t_n - \hat{y}_n$ to confirm zero mean ($\bar{r} = 0.6797 \approx 0$) and homoscedasticity. |
| **Unit 2** | **Regularization (Ridge)** | $L_2$ shrinkage parameter | Theoretical and experimental analysis of $\lambda \|\mathbf{w}\|^2$ penalty to counter multicollinearity. |

---

## 4. Dataset Architecture & Exploratory Data Analysis (EDA)

### 4.1 Dataset Structure & Academic Integrity
- **Origin:** Specifically synthesized project dataset for demonstrating Least Squares regression without violating student privacy.
- **Total Records:** $N = 200$ samples.
- **Partition:** 160 Training samples (80%), 40 Testing samples (20%).
- **Data Quality:** 0 null values, 0 duplicate rows, 100% within legitimate domain bounds.

| Variable Name | Mathematical Symbol | Domain / Range | Measurement Unit | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `Study_Hours` | $x_1$ | $[0.5, 14.0]$ | Hours / day | Continuous Feature 1 |
| `Attendance` | $x_2$ | $[35.0, 100.0]$ | Percentage (%) | Continuous Feature 2 |
| `Assignment_Marks` | $x_3$ | $[15.0, 100.0]$ | Score $[0, 100]$ | Continuous Feature 3 |
| `Internal_Assessment_Marks` | $x_4$ | $[20.0, 100.0]$ | Score $[0, 100]$ | Continuous Feature 4 |
| `Final_Exam_Marks` | $t$ | $[0.0, 100.0]$ | Score $[0, 100]$ | **Target Response ($t$)** |

### 4.2 Descriptive Statistical Summary ($N=200$)

| Feature | Mean | Std Dev | Min | 25% (Q1) | 50% (Median) | 75% (Q3) | Max | Skewness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Study Hours ($x_1$)** | 6.32 | 3.25 | 0.50 | 3.68 | 6.35 | 8.82 | 11.90 | +0.09 |
| **Attendance ($x_2$)** | 70.26 | 17.58 | 35.00 | 56.40 | 71.40 | 85.23 | 99.40 | -0.13 |
| **Assignment Marks ($x_3$)** | 61.66 | 24.58 | 15.00 | 41.25 | 62.45 | 85.62 | 100.00 | -0.07 |
| **Internal Assessment ($x_4$)** | 58.30 | 22.31 | 20.00 | 39.80 | 57.65 | 78.87 | 99.20 | +0.15 |
| **Final Exam Marks ($t$)** | 76.12 | 15.27 | 28.40 | 66.85 | 77.85 | 88.35 | 100.00 | -0.34 |

### 4.3 Pearson Correlation Analysis
The linear correlation matrix $\mathbf{R} \in \mathbb{R}^{5 \times 5}$ reveals key insights into feature relationships:

$$\begin{bmatrix} 
& \text{Study} & \text{Attendance} & \text{Assignment} & \text{Internal} & \text{Final} \\
\text{Study} & 1.000 & -0.026 & -0.047 & -0.032 & \mathbf{+0.694} \\
\text{Attendance} & -0.026 & 1.000 & +0.012 & -0.019 & \mathbf{+0.080} \\
\text{Assignment} & -0.047 & +0.012 & 1.000 & +0.025 & \mathbf{+0.423} \\
\text{Internal} & -0.032 & -0.019 & +0.025 & 1.000 & \mathbf{+0.476} \\
\text{Final} & \mathbf{+0.694} & \mathbf{+0.080} & \mathbf{+0.423} & \mathbf{+0.476} & 1.000
\end{bmatrix}$$

**Key Observations:**
1. **Study Hours** has the highest individual linear correlation with Final Exam Marks ($r = +0.694$), showing that self-study is the single strongest driver of exam success.
2. **Internal Assessment Marks** ($r = +0.476$) and **Assignment Marks** ($r = +0.423$) have strong, statistically significant positive correlations with final performance.
3. The pairwise correlations between the input features themselves are near zero ($|r| < 0.05$). This indicates **absence of multicollinearity**, guaranteeing that the Gram Matrix $\mathbf{X}^T \mathbf{X}$ is well-conditioned and non-singular.

---

## 5. Mathematical Formulation & Theoretical Foundations

### 5.1 Multiple Linear Regression Model
We formulate the hypothesis function as an affine linear combination of the $D=4$ academic features:
$$\hat{t}(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4$$

To simplify matrix calculus, we define a dummy bias input $x_0 \equiv 1$ and express the prediction as an inner product:
$$\hat{t}(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{D} w_j x_j = \mathbf{w}^T \mathbf{x}$$
where:
- $\mathbf{x} = [1, x_1, x_2, x_3, x_4]^T \in \mathbb{R}^{D+1}$ is the augmented feature vector.
- $\mathbf{w} = [w_0, w_1, w_2, w_3, w_4]^T \in \mathbb{R}^{D+1}$ is the parameter weight vector to be estimated.

### 5.2 Augmented Design Matrix
For a training dataset of $N=160$ observations, we construct the Design Matrix $\mathbf{X}$ by stacking all augmented feature rows:
$$\mathbf{X} = \begin{bmatrix} 
1 & x_{11} & x_{12} & x_{13} & x_{14} \\ 
1 & x_{21} & x_{22} & x_{23} & x_{24} \\ 
\vdots & \vdots & \vdots & \vdots & \vdots \\ 
1 & x_{N1} & x_{N2} & x_{N3} & x_{N4} 
\end{bmatrix} \in \mathbb{R}^{N \times (D+1)} = \mathbb{R}^{160 \times 5}$$

The target vector is:
$$\mathbf{t} = [t_1, t_2, \dots, t_N]^T \in \mathbb{R}^{N} = \mathbb{R}^{160}$$

### 5.3 Sum-of-Squares Error (SSE) Function
The prediction residual for student $n$ is:
$$r_n = t_n - \hat{y}_n = t_n - \mathbf{w}^T \mathbf{x}_n$$

The Sum-of-Squares Error objective function $E(\mathbf{w})$ is defined as:
$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} \left( \mathbf{w}^T \mathbf{x}_n - t_n \right)^2 = \frac{1}{2} \|\mathbf{X}\mathbf{w} - \mathbf{t}\|^2$$

Expanding the matrix norm:
$$E(\mathbf{w}) = \frac{1}{2} (\mathbf{X}\mathbf{w} - \mathbf{t})^T (\mathbf{X}\mathbf{w} - \mathbf{t}) = \frac{1}{2} \left( \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w} - 2 \mathbf{t}^T \mathbf{X} \mathbf{w} + \mathbf{t}^T \mathbf{t} \right)$$

### 5.4 Analytical Derivation of the Normal Equations
Because $E(\mathbf{w})$ is a quadratic function of $\mathbf{w}$ and the Hessian matrix $\nabla_{\mathbf{w}}^2 E(\mathbf{w}) = \mathbf{X}^T \mathbf{X}$ is positive semi-definite, $E(\mathbf{w})$ is strictly convex. A global minimum occurs where the gradient vector equals zero:

$$\nabla_{\mathbf{w}} E(\mathbf{w}) = \mathbf{X}^T \mathbf{X} \mathbf{w} - \mathbf{X}^T \mathbf{t} = \mathbf{0}$$

Rearranging gives the celebrated **Normal Equations**:
$$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$$

- **Gram Matrix $\mathbf{X}^T \mathbf{X} \in \mathbb{R}^{5 \times 5}$:** Captures the inner products between all pairs of features.
- **Moment Vector $\mathbf{X}^T \mathbf{t} \in \mathbb{R}^{5 \times 1}$:** Captures the projection of the target marks onto the feature space.

### 5.5 Probabilistic Equivalence: Maximum Likelihood Estimation (MLE)
In Unit 2, we establish the probabilistic justification of Least Squares. Assume that the true target $t_n$ is generated by a deterministic linear mapping plus additive Gaussian noise:
$$t_n = \mathbf{w}^T \mathbf{x}_n + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma^2)$$

The conditional probability density of observing target $t_n$ given input $\mathbf{x}_n$ is:
$$p(t_n \mid \mathbf{x}_n, \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(t_n - \mathbf{w}^T \mathbf{x}_n)^2}{2\sigma^2} \right)$$

Assuming observations are independent and identically distributed (i.i.d.), the likelihood function is:
$$L(\mathbf{w}, \sigma^2) = \prod_{n=1}^N p(t_n \mid \mathbf{x}_n, \mathbf{w}, \sigma^2)$$

Taking the natural logarithm:
$$\ln L(\mathbf{w}, \sigma^2) = -\frac{N}{2} \ln(2\pi) - \frac{N}{2} \ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{n=1}^N (t_n - \mathbf{w}^T \mathbf{x}_n)^2$$

Maximizing $\ln L(\mathbf{w}, \sigma^2)$ with respect to $\mathbf{w}$ requires maximizing the negative sum-of-squares term, which is strictly identical to **minimizing the Sum-of-Squares Error $E(\mathbf{w})$**:
$$\arg\max_{\mathbf{w}} \ln L(\mathbf{w}, \sigma^2) = \arg\min_{\mathbf{w}} \frac{1}{2} \sum_{n=1}^N (t_n - \mathbf{w}^T \mathbf{x}_n)^2 = \mathbf{w}_{\text{OLS}}$$

$$\therefore \mathbf{w}_{\text{MLE}} \equiv \mathbf{w}_{\text{OLS}} \quad \text{under Gaussian noise.}$$

### 5.6 Numerical Stability: `solve` vs `inv`
Mathematically, the weight vector can be written as:
$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{t}$$

However, in computational software engineering, computing the explicit matrix inverse $(\mathbf{X}^T \mathbf{X})^{-1}$ using `numpy.linalg.inv` is strictly avoided due to:
1. **Numerical Instability:** Matrix inversion has high condition number sensitivity; slight collinearities amplify roundoff errors.
2. **Computational Inefficiency:** Explicit inversion has $O(D^3)$ computational cost and performs unnecessary operations.

Instead, our implementation uses `numpy.linalg.solve(XtX, Xty)`, which leverages **LU decomposition with partial pivoting** (or Cholesky decomposition for symmetric positive definite systems), guaranteeing maximum floating-point precision.

---

## 6. Software Architecture & Implementation

### 6.1 Object-Oriented Architecture
The model is implemented in `src/least_squares.py` and embedded in the Colab notebook as `LeastSquaresRegressor`:

```python
class LeastSquaresRegressor:
    """
    Ordinary Least Squares Multiple Linear Regression from scratch.
    Estimates regression weights w via analytical Normal Equations: (X^TX)w = X^Tt.
    """
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or [
            'Study_Hours', 'Attendance', 'Assignment_Marks', 'Internal_Assessment_Marks'
        ]
        self.weights_ = None
        self.intercept_ = None
        self.coefficients_ = None
        self.training_sse_ = None
        self.is_fitted_ = False
```

### 6.2 The 10-Step Fitting Procedure (`fit`)
1. Validate numeric types, non-null values, and 2D shape compatibility.
2. Construct design matrix $\mathbf{X}_{\text{design}} = [\mathbf{1} \mid \mathbf{X}]$ by prepending ones.
3. Compute the $5 \times 5$ Gram Matrix $\mathbf{X}^T \mathbf{X}$.
4. Compute the $5 \times 1$ Moment Vector $\mathbf{X}^T \mathbf{t}$.
5. Solve $(\mathbf{X}^T \mathbf{X}) \mathbf{w} = \mathbf{X}^T \mathbf{t}$ via `np.linalg.solve`.
6. Extract intercept $w_0 = \mathbf{w}[0]$ and feature weights $w_{1:4} = \mathbf{w}[1:]$.
7. Set `self.is_fitted_ = True`.
8. Generate in-sample training predictions $\hat{\mathbf{y}}_{\text{train}} = \mathbf{X}_{\text{design}} \mathbf{w}$.
9. Compute training residuals $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$.
10. Calculate and store training Sum-of-Squares Error (SSE).

---

## 7. Experimental Results & Verification

### 7.1 Normal Equations System Components

#### 1. Gram Matrix $\mathbf{X}^T \mathbf{X} \in \mathbb{R}^{5 \times 5}$:
$$\mathbf{X}^T \mathbf{X} = \begin{bmatrix}
160.00 & 1008.00 & 11190.00 & 9933.40 & 9221.60 \\
1008.00 & 8029.24 & 70239.76 & 61625.54 & 58037.19 \\
11190.00 & 70239.76 & 833284.98 & 687006.03 & 651510.91 \\
9933.40 & 61625.54 & 687006.03 & 711939.70 & 570538.03 \\
9221.60 & 58037.19 & 651510.91 & 570538.03 & 603047.44
\end{bmatrix}$$

#### 2. Moment Vector $\mathbf{X}^T \mathbf{t} \in \mathbb{R}^{5 \times 1}$:
$$\mathbf{X}^T \mathbf{t} = \begin{bmatrix}
12124.20 \\
81797.22 \\
852260.48 \\
774625.93 \\
721383.44
\end{bmatrix}$$

### 7.2 Learned Parameter Weights

Solving the linear system yields the exact optimal parameter vector:

| Parameter | Feature Name | Learned Weight ($w_j$) | Physical & Academic Interpretation |
| :--- | :--- | :--- | :--- |
| $w_0$ | **Intercept (Bias)** | **+11.4137** | Baseline expected mark when all study hours, attendance, assignments, and internals are zero. |
| $w_1$ | **Study Hours** | **+3.4112** | Every additional hour of daily self-study increases final exam score by **~3.41 marks**, holding other features constant. |
| $w_2$ | **Attendance** | **+0.1040** | Every 10% increase in class attendance contributes **~1.04 marks** to the final examination. |
| $w_3$ | **Assignment Marks** | **+0.2792** | Every 10-mark improvement in assignments contributes **~2.79 marks** to the final examination. |
| $w_4$ | **Internal Assessment** | **+0.3168** | Every 10-mark improvement in internal tests contributes **~3.17 marks** to the final examination. |

### 7.3 Learned Regression Equation
$$\mathbf{\hat{y}} = 11.4137 + 3.4112 \cdot (\text{Study\_Hours}) + 0.1040 \cdot (\text{Attendance}) + 0.2792 \cdot (\text{Assignment}) + 0.3168 \cdot (\text{Internal})$$

### 7.4 Model Evaluation & Scikit-Learn Verification

The model's predictions were independently verified against Scikit-Learn on both the training partition ($N=160$) and the unseen testing partition ($N=40$):

| Evaluation Metric | Manual Implementation | Scikit-Learn Reference | Absolute Difference | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Training MSE** | `16.9847` | `16.9847` | `0.000000` | **Exact Match [PASS]** |
| **Testing MSE (Primary)** | `16.6786` | `16.6786` | `0.000000` | **Exact Match [PASS]** |
| **Training RMSE** | `4.1212` marks | `4.1212` marks | `0.000000` | **Exact Match [PASS]** |
| **Testing RMSE** | `4.0839` marks | `4.0839` marks | `0.000000` | **Exact Match [PASS]** |
| **Training MAE** | `3.2428` marks | `3.2428` marks | `0.000000` | **Exact Match [PASS]** |
| **Testing MAE** | `3.3422` marks | `3.3422` marks | `0.000000` | **Exact Match [PASS]** |
| **Training $R^2$** | `0.9222` (92.22%) | `0.9222` (92.22%) | `0.000000` | **Exact Match [PASS]** |
| **Testing $R^2$ (Primary)**| **`0.9413` (94.13%)** | **`0.9413` (94.13%)** | `0.000000` | **Exact Match [PASS]** |

**Interpretation:**
- An out-of-sample $R^2$ of **0.9413** indicates that **94.13% of the total variance in unseen students' final examination marks is explained** by the 4 academic features.
- Testing MSE ($16.68$) is lower than Training MSE ($16.98$), demonstrating outstanding generalization with **zero overfitting**.
- An RMSE of $4.08$ marks on a $0-100$ scale demonstrates that typical model predictions fall within $\pm 4$ marks of actual exam performance.

---

## 8. Residual Diagnostics & Assumption Checks

Linear regression relies on key Gauss-Markov assumptions regarding residuals $r_n = t_n - \hat{y}_n$:

### 8.1 Summary of Residual Statistics (Testing Set, $N=40$)
- **Mean Residual ($\bar{r}$):** `+0.6797` (Very close to $0.0$, confirming absence of systematic directional bias).
- **Residual Standard Deviation ($s_r$):** `4.0270` marks.
- **Residual Variance ($s_r^2$):** `16.2166`.
- **Minimum Residual:** `-7.41` marks.
- **Maximum Residual:** `+8.80` marks.

### 8.2 Diagnostic Plot Interpretations
1. **Actual vs. Predicted Scatter:** Data points cluster tightly along the $45^\circ$ ideal line ($y = x$) across the entire score spectrum ($30$ to $100$ marks), demonstrating linear calibration.
2. **Residuals vs. Fitted Values:** Residuals are randomly distributed around the horizontal zero-line ($r = 0$) with uniform dispersion across all prediction levels. This validates the **Homoscedasticity assumption** (constant variance).
3. **Residual Distribution:** The histogram and KDE display a bell-shaped, symmetric profile centered near zero, validating the **Gaussian noise assumption** $\epsilon \sim \mathcal{N}(0, \sigma^2)$ required for Maximum Likelihood equivalence.

---

## 9. Step-by-Step Viva Arithmetic Walkthrough

To demonstrate exact manual calculation during examination queries:

### Sample Student Profile:
- **Study Hours ($x_1$):** $7.5$ hours/day
- **Attendance ($x_2$):** $80.0\%$
- **Assignment Marks ($x_3$):** $70.0$ / 100
- **Internal Assessment ($x_4$):** $65.0$ / 100

### Step-by-Step Dot Product $\mathbf{w}^T \mathbf{x}$:
$$\begin{aligned}
\hat{y} &= w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 \\
&= 11.4137 + (3.4112 \times 7.5) + (0.1040 \times 80.0) + (0.2792 \times 70.0) + (0.3168 \times 65.0) \\
&= 11.4137 + 25.5838 + 8.3209 + 19.5473 + 20.5946 \\
&= \mathbf{85.4604} \text{ Marks}
\end{aligned}$$

**Predicted Final Examination Marks:** **85.46 / 100**  
**Academic Standing:** **First Class with Distinction**

---

## 10. Advanced Academic Extensions

### 10.1 Unit 1 Extension: Polynomial Curve Fitting
To demonstrate the Unit 1 syllabus topic of polynomial curve fitting and model complexity, we extended the feature space by appending a quadratic term for study hours:
$$\hat{t}_{\text{poly}} = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 + w_5 (x_1^2)$$

| Model Architecture | Parameter Count | Training MSE | Testing MSE | Training $R^2$ | Testing $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Least Squares (Primary)** | **5 parameters** | **16.98** | **16.68** | **0.9222** | **0.9413** |
| **Polynomial Extension ($x_1^2$)** | 6 parameters | 16.63 | 16.24 | 0.9238 | 0.9428 |

**Analysis:** The quadratic term yields only a minor gain in $R^2$ ($+0.0015$), confirming that the underlying relationship is overwhelmingly linear and that higher-order polynomial expansion risks unnecessary complexity without meaningful generalization improvement.

### 10.2 Unit 2 Extension: Ridge Regression ($L_2$ Regularization)
In cases where features exhibit high multicollinearity, the Gram matrix $\mathbf{X}^T \mathbf{X}$ can become nearly singular (ill-conditioned), causing weight estimates to explode. Ridge Regression introduces an $L_2$ penalty:
$$E_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (t_n - \mathbf{w}^T \mathbf{x}_n)^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

Setting the gradient to zero yields the regularized Normal Equations:
$$(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}) \mathbf{w} = \mathbf{X}^T \mathbf{t} \implies \mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{t}$$

Adding $\lambda \mathbf{I}$ (where $\lambda > 0$) guarantees that the matrix $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})$ is strictly positive definite and always invertible, preventing numeric instability while shrinking parameter variance.

---

## 11. Viva Voce Revision Guide (Key Examination Q&A)

1. **Why is this problem categorized under Regression rather than Classification?**  
   *Answer:* The target variable (*Final Examination Marks*) is continuous and metric-ordered on $[0, 100]$. Classification is reserved for discrete qualitative category assignments (e.g., Pass vs. Fail).

2. **Why do we prepend a column of ones to the Design Matrix?**  
   *Answer:* It enables the intercept/bias term $w_0$ to be estimated as an intrinsic element of the vector dot product $\mathbf{w}^T \mathbf{x}$. Without this column, the hyperplane would be forced through the origin $(0,0,0,0,0)$.

3. **Why do we square the residuals instead of taking absolute values?**  
   *Answer:* Squaring ensures penalization regardless of sign, produces a continuously differentiable convex loss function with a unique global minimum, and naturally links to Maximum Likelihood Estimation under Gaussian noise.

4. **Why use `numpy.linalg.solve()` instead of `numpy.linalg.inv()`?**  
   *Answer:* Matrix inversion is computationally expensive ($O(D^3)$) and numerically unstable. `np.linalg.solve()` uses LU/Cholesky factorization, minimizing floating-point roundoff errors and condition number degradation.

5. **What does an $R^2$ score of 0.9413 mean?**  
   *Answer:* It means 94.13% of the total variance in unseen final exam scores is directly explained by the model's four academic features, leaving only 5.87% to unexplained random error.

---

## 12. Strengths, Limitations, and Future Scope

### 12.1 Strengths
1. **100% Mathematical Transparency:** Completely derived and implemented from scratch without high-level black-box libraries.
2. **High Generalization Accuracy:** $R^2 = 0.9413$ and $\text{RMSE} = 4.08$ on completely unseen student profiles.
3. **Resilient Architecture:** Multi-tier data loader and headless automated testing ensure zero-crash execution across any machine or Colab environment.

### 12.2 Limitations
1. **Assumption of Linearity:** Assumes strictly linear feature contributions without complex interactive coupling (e.g., attendance modulating study hour efficiency).
2. **Synthetic Dataset:** Developed for pedagogical course demonstration; real student behaviors may exhibit greater noise and institutional variability.

### 12.3 Future Scope
1. **Real Institution Integration:** Validating on real SRM University student cohorts across diverse engineering disciplines.
2. **Interactive Feature Crosses:** Modeling interaction terms such as $\text{Attendance} \times \text{Study\_Hours}$.
3. **Automated Warning System:** Embedding the model into student academic portals to trigger automated study advisories for students predicted below pass thresholds.

---

## 13. Conclusion

This project successfully demonstrates the theoretical formulation, software engineering implementation, and rigorous validation of **Student Marks Prediction Using Least Squares**. By bridging the theoretical foundations of Units 1 & 2 of the SRM University Machine Learning syllabus with an end-to-end Python implementation, the project provides a benchmark of academic and computational rigor.

---

## 14. References

1. Bishop, Christopher M. *Pattern Recognition and Machine Learning*. Springer, 2006. (Chapter 1: "Introduction & Polynomial Curve Fitting", Chapter 3: "Linear Models for Regression").
2. Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer, 2nd ed., 2009.
3. Strang, Gilbert. *Linear Algebra and Its Applications*. Cengage Learning, 4th ed., 2006. (Normal Equations and Projections).
4. SRM Institute of Science and Technology. *Department of Computing Technologies — Machine Learning Course Curriculum (Units 1 & 2)*.
5. NumPy Documentation: `numpy.linalg.solve` — Numerical Linear Algebra Routines (https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html).
