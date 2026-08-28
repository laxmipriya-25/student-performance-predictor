"""
generate_dataset.py
--------------------
This script CREATES a small, realistic, synthetic student performance
dataset and saves it as data/student_performance.csv

You only need to run this ONCE (or whenever you want to regenerate the data).
If you already have your own real dataset, just place it at
data/student_performance.csv with the same column names and skip this script.

Columns created:
- study_hours              : Average hours the student studies per day   (numeric, ~0-10)
- attendance_percentage    : Class attendance percentage                 (numeric, 0-100)
- previous_score           : Score obtained in the previous exam         (numeric, 0-100)
- sleep_hours              : Average hours of sleep per day              (numeric, ~4-10)
- assignment_completion    : Percentage of assignments completed         (numeric, 0-100)
- extracurricular          : Whether student takes part in extracurricular
                              activities                                 (categorical: "Yes"/"No")
- final_score              : TARGET - Final exam/performance score       (numeric, 0-100)

The final_score is generated using a realistic (but not perfectly linear)
formula based on the other features, plus random noise, so the models have
real patterns to learn instead of pure randomness.
"""

import numpy as np
import pandas as pd

# Set a random seed so the dataset is reproducible every time this script runs
np.random.seed(42)

# Number of student records to generate
N_STUDENTS = 500

# ---------------------------------------------------------------------------
# 1. Generate each feature with realistic ranges and distributions
# ---------------------------------------------------------------------------

# Study hours per day: most students study between 0 and 8 hours (normal-ish distribution)
study_hours = np.clip(np.random.normal(loc=4.0, scale=1.8, size=N_STUDENTS), 0, 10)

# Attendance percentage: most students have decent attendance
attendance_percentage = np.clip(np.random.normal(loc=80, scale=12, size=N_STUDENTS), 40, 100)

# Previous exam score: fairly spread out
previous_score = np.clip(np.random.normal(loc=65, scale=15, size=N_STUDENTS), 20, 100)

# Sleep hours per day: most students sleep 5-9 hours
sleep_hours = np.clip(np.random.normal(loc=6.8, scale=1.2, size=N_STUDENTS), 3, 10)

# Assignment completion percentage
assignment_completion = np.clip(np.random.normal(loc=75, scale=18, size=N_STUDENTS), 0, 100)

# Extracurricular activity: Yes/No (categorical), ~40% participate
extracurricular = np.random.choice(["Yes", "No"], size=N_STUDENTS, p=[0.4, 0.6])
extracurricular_numeric = (extracurricular == "Yes").astype(int)  # used only for score formula

# ---------------------------------------------------------------------------
# 2. Create the target variable (final_score) using a realistic formula
#    - More study hours, attendance, previous score, sleep, and assignment
#      completion generally increase the final score.
#    - Extracurricular activity has a small positive effect (well-rounded
#      students), but too little sleep or study can hurt performance.
#    - Random noise is added to simulate real-world unpredictability.
# ---------------------------------------------------------------------------
noise = np.random.normal(loc=0, scale=5, size=N_STUDENTS)

final_score = (
    (study_hours * 4.5)
    + (attendance_percentage * 0.25)
    + (previous_score * 0.30)
    + (sleep_hours * 1.2)
    + (assignment_completion * 0.15)
    + (extracurricular_numeric * 2.0)
    + noise
)

# Clip final scores to a realistic 0-100 range
final_score = np.clip(final_score, 0, 100)

# ---------------------------------------------------------------------------
# 3. Assemble the DataFrame
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "study_hours": np.round(study_hours, 2),
    "attendance_percentage": np.round(attendance_percentage, 2),
    "previous_score": np.round(previous_score, 2),
    "sleep_hours": np.round(sleep_hours, 2),
    "assignment_completion": np.round(assignment_completion, 2),
    "extracurricular": extracurricular,
    "final_score": np.round(final_score, 2),
})

# ---------------------------------------------------------------------------
# 4. Intentionally introduce a few missing values (to demonstrate that the
#    training script correctly handles missing data, like a real dataset would)
# ---------------------------------------------------------------------------
missing_indices = np.random.choice(df.index, size=10, replace=False)
df.loc[missing_indices, "sleep_hours"] = np.nan

missing_indices_2 = np.random.choice(df.index, size=8, replace=False)
df.loc[missing_indices_2, "assignment_completion"] = np.nan

# ---------------------------------------------------------------------------
# 5. Save to CSV
# ---------------------------------------------------------------------------
output_path = "data/student_performance.csv"
df.to_csv(output_path, index=False)

print(f"Dataset generated successfully with {len(df)} rows.")
print(f"Saved to: {output_path}")
print("\nPreview:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
