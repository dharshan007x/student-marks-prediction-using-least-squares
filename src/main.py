"""
main.py
=======
Command-line interface for the Student Marks Prediction project.

When executed:
    python src/main.py

This script demonstrates the complete Least Squares workflow:
    1.  Load the dataset
    2.  Validate the dataset
    3.  Split into training and testing sets
    4.  Build the design matrix
    5.  Train the Least Squares model (Normal Equations)
    6.  Display XᵀX, Xᵀy, learned coefficients
    7.  Evaluate on training and testing data
    8.  Predict marks for an example student
    9.  Save visualizations to outputs/
"""

import sys
import os
import numpy as np

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    load_dataset, validate_dataset, format_validation_report,
    get_features_and_target, split_data, FEATURE_COLUMNS, TARGET_COLUMN,
)
from src.least_squares import LeastSquaresRegressor
from src.evaluation import full_evaluation, residual_table
from src.visualizations import (
    plot_all_features_vs_target, plot_actual_vs_predicted,
    plot_residuals, save_model_statistics,
)
from src.utils import (
    get_data_path, get_output_path, print_header, print_subheader,
    format_equation, format_matrix, format_vector, predict_single_student,
)


def main():
    # ==================================================================
    # 1. LOAD DATASET
    # ==================================================================
    print_header("STUDENT MARKS PREDICTION USING LEAST SQUARES")
    print("  Predicting Final Examination Marks Using")
    print("  Multiple Linear Least Squares Regression")
    print("=" * 60)

    data_path = get_data_path()
    print(f"\nLoading dataset from: {data_path}")
    df = load_dataset(data_path)
    print(f"Dataset loaded successfully: {df.shape[0]} rows * {df.shape[1]} columns")

    # ==================================================================
    # 2. VALIDATE DATASET
    # ==================================================================
    print_header("DATASET VALIDATION")
    report = validate_dataset(df)
    print(format_validation_report(report))

    # ==================================================================
    # 3. DATASET SUMMARY
    # ==================================================================
    print_header("DATASET SUMMARY")
    print(f"  Total Students     : {len(df)}")
    print(f"  Input Features (4) : {FEATURE_COLUMNS}")
    print(f"  Target Variable    : {TARGET_COLUMN}")
    print()
    print("  Descriptive Statistics:")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().round(2).to_string())

    # ==================================================================
    # 4. FEATURE EXTRACTION & TRAIN/TEST SPLIT
    # ==================================================================
    print_header("TRAIN / TEST SPLIT")
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    print(f"  Total Samples   : {len(X)}")
    print(f"  Training Samples: {len(X_train)}  (80%)")
    print(f"  Testing Samples : {len(X_test)}   (20%)")
    print()
    print("  Importance of Generalization:")
    print("  The model is fitted ONLY on training data.")
    print("  Testing data remains unseen during coefficient estimation.")
    print("  This ensures that evaluation reflects the model's ability")
    print("  to generalize to new, unseen student records.")

    # ==================================================================
    # 5. BUILD DESIGN MATRIX
    # ==================================================================
    print_header("DESIGN MATRIX")
    model = LeastSquaresRegressor(feature_names=FEATURE_COLUMNS)
    X_design_train = model.get_design_matrix(X_train)

    print(f"  Feature matrix X_train shape : {X_train.shape}")
    print(f"  Design matrix  shape         : {X_design_train.shape}")
    print()
    print("  The design matrix prepends a column of 1s for the intercept:")
    print()
    print(format_matrix(X_design_train, "  X (Design Matrix)", max_rows=5))

    # ==================================================================
    # 6. TRAIN MODEL (NORMAL EQUATIONS)
    # ==================================================================
    print_header("MODEL TRAINING - NORMAL EQUATIONS")
    model.fit(X_train, y_train)

    details = model.get_training_details()

    print_subheader("X'X  (Gram Matrix)")
    print(format_matrix(details["XtX"], "  X'X", max_rows=5, decimals=2))

    print_subheader("X'y  (Moment Vector)")
    print(format_vector(details["Xty"], "  X'y"))

    print_subheader("Learned Coefficient Vector w")
    print(format_vector(details["weights"], "  w"))

    print_subheader("Intercept")
    print(f"  w0 = {model.intercept_:.4f}")

    print_subheader("Regression Coefficients")
    for name, coef in zip(FEATURE_COLUMNS, model.coefficients_):
        print(f"  {name:30s} : {coef:+.4f}")

    # ==================================================================
    # 7. MODEL EQUATION
    # ==================================================================
    print_header("LEARNED REGRESSION EQUATION")
    eq = format_equation(model.intercept_, model.coefficients_,
                         FEATURE_COLUMNS, "Predicted Final Exam Marks")
    print(f"\n{eq}")

    # ==================================================================
    # 8. TRAINING EVALUATION
    # ==================================================================
    print_header("TRAINING SET EVALUATION")
    train_eval = full_evaluation(y_train, model.predict(X_train))
    print(f"  Sum-of-Squares Error : {model.training_error_:.4f}")
    print(f"  MSE                  : {train_eval['MSE']:.4f}")
    print(f"  RMSE                 : {train_eval['RMSE']:.4f}")
    print(f"  MAE                  : {train_eval['MAE']:.4f}")
    print(f"  R²                   : {train_eval['R2']:.4f}")

    # ==================================================================
    # 9. TESTING SET EVALUATION (PRIMARY RESULT)
    # ==================================================================
    print_header("TESTING SET EVALUATION  ** PRIMARY RESULT **")
    y_pred_test = model.predict(X_test)
    test_eval = full_evaluation(y_test, y_pred_test)
    test_sse = model.sum_squared_error(X_test, y_test)
    print(f"  Sum-of-Squares Error : {test_sse:.4f}")
    print(f"  MSE                  : {test_eval['MSE']:.4f}")
    print(f"  RMSE                 : {test_eval['RMSE']:.4f}")
    print(f"  MAE                  : {test_eval['MAE']:.4f}")
    print(f"  R²                   : {test_eval['R2']:.4f}")

    # ==================================================================
    # 10. RESIDUAL ANALYSIS
    # ==================================================================
    print_header("RESIDUAL ANALYSIS (Test Set)")
    res_stats = test_eval["residual_stats"]
    print(f"  Mean Residual   : {res_stats['mean']:.4f}")
    print(f"  Std Residual    : {res_stats['std']:.4f}")
    print(f"  Min Residual    : {res_stats['min']:.4f}")
    print(f"  Max Residual    : {res_stats['max']:.4f}")
    print(f"  Median Residual : {res_stats['median']:.4f}")

    print_subheader("Top 5 Largest Absolute Residuals")
    res_tbl = residual_table(y_test, y_pred_test, n_largest=5)
    print(res_tbl.to_string(index=False))

    # ==================================================================
    # 11. PREDICT FOR A NEW STUDENT
    # ==================================================================
    print_header("PREDICTION FOR A NEW STUDENT")
    example = predict_single_student(
        model,
        study_hours=7.5,
        attendance=85.0,
        assignment_marks=72.0,
        internal_marks=68.0,
    )
    print("  Input Values:")
    for k, v in example["input_values"].items():
        print(f"    {k:30s} : {v}")
    print()
    print("  Step-by-step Calculation:")
    print(f"    Feature vector: {example['feature_vector']}")
    print(f"    With intercept: [1.0, {', '.join(f'{v:.1f}' for v in example['feature_vector'])}]")
    print()
    print("  Contribution Breakdown:")
    total = 0.0
    for name, contribution in example["contribution_breakdown"].items():
        print(f"    {name:35s} : {contribution:+.4f}")
        total += contribution
    print(f"    {'':35s}   {'-' * 10}")
    print(f"    {'Total':35s} : {total:.4f}")
    print()
    print(f"  >> Predicted Final Examination Marks: {example['predicted_marks']:.2f}")

    # ==================================================================
    # 12. SAVE VISUALIZATIONS
    # ==================================================================
    print_header("SAVING VISUALIZATIONS")

    path1 = get_output_path("feature_relationships.png")
    plot_all_features_vs_target(df, FEATURE_COLUMNS, save_path=path1)
    print(f"  [SAVED] {path1}")

    path2 = get_output_path("actual_vs_predicted.png")
    plot_actual_vs_predicted(y_test, y_pred_test, save_path=path2)
    print(f"  [SAVED] {path2}")

    path3 = get_output_path("residual_plot.png")
    plot_residuals(y_test, y_pred_test, save_path=path3)
    print(f"  [SAVED] {path3}")

    path4 = get_output_path("model_statistics.csv")
    coef_dict = {"intercept": model.intercept_}
    for name, c in zip(FEATURE_COLUMNS, model.coefficients_):
        coef_dict[name] = c
    save_model_statistics(train_eval, test_eval, coef_dict, path4)
    print(f"  [SAVED] {path4}")

    # ==================================================================
    # DONE
    # ==================================================================
    print_header("COMPLETE")
    print("  All outputs saved to the outputs/ directory.")
    print("  Run the Streamlit dashboard:")
    print("    streamlit run app/streamlit_app.py")
    print()


if __name__ == "__main__":
    main()
