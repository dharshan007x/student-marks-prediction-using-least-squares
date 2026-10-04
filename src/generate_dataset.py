"""
generate_dataset.py
====================
Generates a synthetic student performance dataset for the
Student Marks Prediction Using Least Squares project.

IMPORTANT — Academic Integrity Notice:
    This dataset is SYNTHETIC / PROJECT DATA created specifically for
    demonstrating the Least Squares regression method.  It is NOT
    collected from real students and must NOT be presented as real-world
    collected student data.

The dataset contains 200 student records with the following columns:
    - Study_Hours           (float, 0–12 hours per day)
    - Attendance            (float, 0–100 percent)
    - Assignment_Marks      (float, 0–100)
    - Internal_Assessment_Marks  (float, 0–100)
    - Final_Exam_Marks      (float, 0–100)

Final_Exam_Marks is generated as a linear combination of the four
input features plus Gaussian noise, ensuring a realistic but
learnable relationship suitable for Least Squares regression.
"""

import numpy as np
import pandas as pd
import os


def generate_student_dataset(
    n_students: int = 200,
    random_state: int = 42,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic student marks dataset.

    The underlying data-generating process is:

        Final_Exam_Marks = 5.0
                         + 3.5 * Study_Hours
                         + 0.15 * Attendance
                         + 0.30 * Assignment_Marks
                         + 0.35 * Internal_Assessment_Marks
                         + ε

    where ε ~ N(0, σ²) with σ = 4.0.

    Parameters
    ----------
    n_students : int
        Number of student records to generate.
    random_state : int
        Seed for reproducibility.
    save_path : str or None
        If provided, saves the DataFrame as a CSV to this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        Study_Hours, Attendance, Assignment_Marks,
        Internal_Assessment_Marks, Final_Exam_Marks
    """
    rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Generate correlated input features
    # ------------------------------------------------------------------
    # Study Hours: 1–12 hrs per day (approximately)
    study_hours = rng.uniform(1.0, 12.0, n_students)

    # Attendance: 40–100% (most students attend reasonably well)
    attendance = rng.uniform(40.0, 100.0, n_students)

    # Assignment Marks: 20–100
    assignment_marks = rng.uniform(20.0, 100.0, n_students)

    # Internal Assessment Marks: 20–100
    internal_marks = rng.uniform(20.0, 100.0, n_students)

    # ------------------------------------------------------------------
    # Generate the target variable using a known linear relationship
    # ------------------------------------------------------------------
    noise = rng.normal(0, 4.0, n_students)

    final_exam_marks = (
        5.0
        + 3.5 * study_hours
        + 0.15 * attendance
        + 0.30 * assignment_marks
        + 0.35 * internal_marks
        + noise
    )

    # Clip to valid range [0, 100]
    final_exam_marks = np.clip(final_exam_marks, 0.0, 100.0)

    # Round for realism
    study_hours = np.round(study_hours, 1)
    attendance = np.round(attendance, 1)
    assignment_marks = np.round(assignment_marks, 1)
    internal_marks = np.round(internal_marks, 1)
    final_exam_marks = np.round(final_exam_marks, 1)

    # ------------------------------------------------------------------
    # Build DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Assignment_Marks": assignment_marks,
        "Internal_Assessment_Marks": internal_marks,
        "Final_Exam_Marks": final_exam_marks,
    })

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}  ({len(df)} records)")

    return df


if __name__ == "__main__":
    # Generate and save the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "student_marks.csv")
    df = generate_student_dataset(save_path=csv_path)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nDescriptive Statistics:\n{df.describe()}")
