# Viva Questions and Answers

## Student Marks Prediction Using Least Squares

---

### 1. What is Machine Learning?

Machine Learning is a field of computer science that gives computers the ability to learn patterns from data without being explicitly programmed. Instead of writing rules manually, the system learns from examples (training data) and improves its performance on a task.

---

### 2. Why is this project supervised learning?

This is supervised learning because the training dataset contains both the input features (Study Hours, Attendance, Assignment Marks, Internal Assessment Marks) and the corresponding target values (Final Examination Marks). The model learns the mapping from inputs to outputs using these labeled examples.

---

### 3. Why is this a regression problem?

It is a regression problem because the target variable — Final Examination Marks — is a **continuous** numerical value (e.g., 72.5, 88.3). In contrast, classification predicts discrete categories (e.g., Pass/Fail). Since we predict a real-valued score, regression is the appropriate formulation.

---

### 4. What are the four input features?

1. **Study Hours** (x₁) — Average daily study hours
2. **Attendance** (x₂) — Class attendance percentage
3. **Assignment Marks** (x₃) — Marks obtained in assignments
4. **Internal Assessment Marks** (x₄) — Internal assessment score

---

### 5. What is the target variable?

The target variable is **Final Examination Marks** (t) — the score a student obtains in the final examination. This is what the model predicts.

---

### 6. Why are Final Examination Marks the target?

Because the assignment specifically requires predicting Final Examination Marks based on the four input features. Final marks represent the overall academic outcome that depends on study effort, attendance, and prior assessments.

---

### 7. What is Least Squares?

Least Squares is a regression method that finds the best-fit linear model by minimizing the **sum of squared differences** between the actual target values and the predicted values. It provides the optimal weight vector that makes the predictions as close as possible to the actual observations.

---

### 8. What is the Sum-of-Squares Error?

The Sum-of-Squares Error is:

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} [y(\mathbf{x}_n, \mathbf{w}) - t_n]^2$$

It measures the total squared prediction error over all training samples. The factor 1/2 is included for mathematical convenience (it cancels when taking the derivative).

---

### 9. Why are the errors squared?

1. **Squaring makes all errors positive** — both overestimates and underestimates contribute positively
2. **Penalizes large errors more** — a residual of 10 contributes 100, not just 10
3. **Differentiable everywhere** — unlike absolute value, the squared function is smooth
4. **Leads to a closed-form solution** — the derivative yields linear equations (Normal Equations)

---

### 10. What is a residual?

A residual is the difference between the actual value and the predicted value:

$$r_n = t_n - y(\mathbf{x}_n, \mathbf{w})$$

If the residual is positive, the actual mark was higher than predicted. If negative, the actual mark was lower than predicted.

---

### 11. What is the design matrix?

The design matrix **X** is an (N × 5) matrix where each row represents a student and each column represents a feature. A column of 1s is prepended to account for the intercept term:

$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{21} & x_{31} & x_{41} \\ 1 & x_{12} & x_{22} & x_{32} & x_{42} \\ \vdots & \vdots & \vdots & \vdots & \vdots \end{bmatrix}$$

---

### 12. Why is the first column of X equal to 1?

The column of 1s allows us to include the **intercept (bias) term** w₀ in the matrix formulation. When we compute X**w**, the first column multiplied by w₀ gives the constant term in the regression equation. Without it, the model would be forced to pass through the origin.

---

### 13. What is the intercept?

The intercept (w₀) is the predicted value when all input features are zero. It represents the baseline prediction. In this model, it is the constant term in the equation:

$$\hat{t} = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4$$

---

### 14. What are regression coefficients?

Regression coefficients (w₁, w₂, w₃, w₄) are the learned parameters that quantify the relationship between each input feature and the target variable. For example, w₁ tells us how much the predicted Final Marks change for a one-unit increase in Study Hours, holding other features constant.

---

### 15. What is the normal equation?

The Normal Equation is:

$$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$$

It is derived by setting the gradient of the Sum-of-Squares Error to zero. Solving this equation gives the optimal weight vector.

---

### 16. Why does Least Squares have a closed-form solution?

Because the Sum-of-Squares Error is a **quadratic function** of the weight vector **w**. Setting its gradient to zero produces a system of linear equations (the Normal Equations), which can be solved directly using linear algebra — no iterative optimization is needed.

---

### 17. Why is numpy.linalg.solve() useful?

`numpy.linalg.solve(A, b)` solves the system Aw = b without explicitly computing the matrix inverse A⁻¹. This is:
- **More numerically stable** — avoids amplifying floating-point errors
- **More efficient** — O(n³) vs potentially worse for explicit inversion
- **Standard practice** in scientific computing

---

### 18. What is Maximum Likelihood Estimation?

Maximum Likelihood Estimation (MLE) is a method that finds the parameter values that **maximize the probability (likelihood)** of observing the given data. It asks: "What parameters make the observed data most probable?"

---

### 19. How is Least Squares related to Maximum Likelihood?

Under the assumption that the observation noise is **Gaussian** (normally distributed):

$$t_n = y(\mathbf{x}_n, \mathbf{w}) + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma^2)$$

Maximizing the log-likelihood is mathematically equivalent to minimizing the Sum-of-Squares Error. Therefore, **Least Squares = Maximum Likelihood under Gaussian noise**.

---

### 20. What Gaussian-noise assumption is involved?

We assume that each observation has additive noise drawn from a Gaussian (normal) distribution with zero mean and variance σ²:

$$\epsilon_n \sim \mathcal{N}(0, \sigma^2)$$

This means the errors are random, centered around zero, and follow a bell curve.

---

### 21. What is MSE?

Mean Squared Error:

$$\text{MSE} = \frac{1}{N} \sum_{n=1}^{N} (t_n - y_n)^2$$

It is the average of the squared prediction errors. It measures how close the predictions are to the actual values on average.

---

### 22. What does a lower MSE mean?

A lower MSE means the model's predictions are closer to the actual values on average. An MSE of 0 would mean perfect predictions (no error at all). In practice, some error is always present due to noise.

---

### 23. What is R²?

R² (Coefficient of Determination):

$$R^2 = 1 - \frac{\sum (t_n - y_n)^2}{\sum (t_n - \bar{t})^2}$$

It measures the proportion of variance in the target variable that is explained by the model. It ranges from negative infinity to 1.

---

### 24. What does R² = 1 mean?

R² = 1 means the model explains 100% of the variance in the target variable — the predictions perfectly match the actual values. This is rare in practice and often indicates overfitting if seen on training data.

---

### 25. Can R² be negative?

Yes. R² can be negative when the model performs **worse than simply predicting the mean** for every sample. This indicates a very poor model. However, for a properly fitted linear regression on training data, R² on the training set will always be non-negative.

---

### 26. Why must testing data be kept separate from training data?

If we evaluate on the same data used for training, the metrics will be **optimistically biased** — the model has already "seen" those examples. Testing on separate data measures the model's ability to **generalize** to new, unseen students, which is the true objective.

---

### 27. What is generalization?

Generalization is the model's ability to perform well on **new, unseen data** — not just the data it was trained on. Good generalization means the model has learned the underlying pattern rather than memorizing the training examples.

---

### 28. How can outliers affect Least Squares?

Outliers have a disproportionate effect because Least Squares **squares** the errors. A single large residual contributes enormously to the total error, potentially pulling the regression line toward the outlier and distorting the coefficients. This is a known limitation of ordinary Least Squares.

---

### 29. What is polynomial curve fitting?

Polynomial curve fitting extends the linear model by adding polynomial terms (e.g., x², x³). This allows the model to capture **nonlinear relationships**. In our project, we optionally add x₁² (Study Hours squared) to demonstrate the concept from Unit 1.

---

### 30. How can a more complex model lead to overfitting?

A more complex model (e.g., high-degree polynomial) has more parameters and can fit the training data very closely — including noise. However, it may perform poorly on new data because it has memorized the training examples rather than learning the general pattern. This is **overfitting**.

---

### 31. What is Ridge Regression?

Ridge Regression adds a **regularization term** to the Least Squares objective:

$$E_{\text{Ridge}} = \frac{1}{2} \sum (y_n - t_n)^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

The λ parameter controls how much the coefficients are penalized for being large. This reduces overfitting and improves stability when features are correlated.

---

### 32. Why is Ridge not the main model in this project?

The assignment specifically requires implementing **Least Squares** (ordinary, unregularized) regression. Ridge Regression is discussed as a **theoretical extension** from Unit 2, but the primary algorithm must remain plain Least Squares with the Normal Equations.

---

### 33. What is the interpretation of the Study Hours coefficient?

The Study Hours coefficient (w₁) represents the expected change in Final Examination Marks for a **one-hour increase** in daily study time, assuming all other features remain constant. For example, if w₁ = 3.41, each additional study hour is associated with approximately 3.41 more marks.

---

### 34. What is the interpretation of the Attendance coefficient?

The Attendance coefficient (w₂) represents the expected change in Final Marks for a **one-percentage-point increase** in attendance, holding other features constant. A positive coefficient means higher attendance is associated with higher marks.

---

### 35. What is the interpretation of the Assignment Marks coefficient?

The Assignment Marks coefficient (w₃) represents the expected change in Final Marks for a **one-mark increase** in assignment score, holding other features constant.

---

### 36. What is the interpretation of the Internal Assessment coefficient?

The Internal Assessment coefficient (w₄) represents the expected change in Final Marks for a **one-mark increase** in internal assessment score, holding other features constant.

---

### 37. Which metric is required by the assignment?

The assignment specifically requires:
1. **Mean Squared Error (MSE)**
2. **R² Score (Coefficient of Determination)**

Both must be calculated and reported on the test set.

---

### 38. Why is MSE more appropriate than classification accuracy for this project?

**Classification accuracy** counts the proportion of correct categorical predictions (e.g., Pass/Fail). Since our target is a **continuous numerical value** (examination marks), accuracy is not meaningful. MSE measures the average squared difference between predicted and actual numerical values, which is the appropriate metric for regression tasks.
