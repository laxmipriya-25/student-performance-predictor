"""
train_model.py
--------------

Main training script for the Student Performance Prediction project.

What it does:
1. Loads the dataset from data/student_performance.csv
2. Handles missing numeric values
3. Encodes extracurricular: Yes/No -> 1/0
4. Splits data into features and target
5. Splits the data into training and testing sets
6. Scales the features using StandardScaler
7. Trains two regression models:
   - Linear Regression
   - Random Forest Regressor
8. Evaluates both models using:
   - MAE
   - MSE
   - RMSE
   - R²
9. Selects the model with the higher R² score
10. Saves the best model, scaler, feature columns, and model name
    into the model/ folder

Run this file with:

    python train_model.py
"""

import os

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------

DATA_PATH = "data/student_performance.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ---------------------------------------------------------------------------
# 2. Handle missing values
#
# For numeric columns, missing values are replaced with the column median.
# Median is used because it is less sensitive to outliers than the mean.
# ---------------------------------------------------------------------------

numeric_cols = [
    "study_hours",
    "attendance_percentage",
    "previous_score",
    "sleep_hours",
    "assignment_completion",
]

print("\nMissing values before cleaning:")
print(df.isnull().sum())

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ---------------------------------------------------------------------------
# 3. Encode the categorical column
#
# extracurricular:
# Yes -> 1
# No  -> 0
# ---------------------------------------------------------------------------

df["extracurricular"] = df["extracurricular"].map({
    "Yes": 1,
    "No": 0
})


# ---------------------------------------------------------------------------
# 4. Define features (X) and target (y)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "study_hours",
    "attendance_percentage",
    "previous_score",
    "sleep_hours",
    "assignment_completion",
    "extracurricular",
]

TARGET_COLUMN = "final_score"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ---------------------------------------------------------------------------
# 5. Train/test split
#
# 80% -> training
# 20% -> testing
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(
    f"\nTraining samples: {X_train.shape[0]}"
)

print(
    f"Testing samples: {X_test.shape[0]}"
)


# ---------------------------------------------------------------------------
# 6. Feature scaling
#
# The scaler is fitted ONLY on training data to avoid data leakage.
# The same scaler is then used on the test data.
# ---------------------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# 7. Train models
# ---------------------------------------------------------------------------

# --- Model 1: Linear Regression ---

lr_model = LinearRegression()

lr_model.fit(
    X_train_scaled,
    y_train
)

lr_predictions = lr_model.predict(X_test_scaled)


# --- Model 2: Random Forest Regressor ---

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

rf_model.fit(
    X_train_scaled,
    y_train
)

rf_predictions = rf_model.predict(X_test_scaled)


# ---------------------------------------------------------------------------
# 8. Evaluate models
# ---------------------------------------------------------------------------

def evaluate_model(name, y_true, y_pred):
    """
    Evaluate a regression model using MAE, MSE, RMSE and R².
    """

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"MAE  : {mae:.3f}")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    return {
        "name": name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


print("\n===================== MODEL EVALUATION =====================")

lr_results = evaluate_model(
    "Linear Regression",
    y_test,
    lr_predictions
)

rf_results = evaluate_model(
    "Random Forest Regressor",
    y_test,
    rf_predictions
)


# ---------------------------------------------------------------------------
# 9. Compare models and select the best model
#
# Higher R² = better fit
# ---------------------------------------------------------------------------

if rf_results["r2"] >= lr_results["r2"]:

    best_model = rf_model
    best_results = rf_results

else:

    best_model = lr_model
    best_results = lr_results


print("\n===================== BEST MODEL =====================")

print(
    f"Selected model: {best_results['name']} "
    f"(R² = {best_results['r2']:.3f})"
)


# ---------------------------------------------------------------------------
# 10. Save the model and preprocessing objects
# ---------------------------------------------------------------------------

# Create model directory if it doesn't already exist

os.makedirs("model", exist_ok=True)


# Save trained model

joblib.dump(
    best_model,
    "model/best_model.pkl"
)


# Save scaler

joblib.dump(
    scaler,
    "model/scaler.pkl"
)


# Save feature column order

joblib.dump(
    FEATURE_COLUMNS,
    "model/feature_columns.pkl"
)


# Save selected model name

joblib.dump(
    best_results["name"],
    "model/model_name.pkl"
)


# ---------------------------------------------------------------------------
# 11. Display saved files
# ---------------------------------------------------------------------------

print("\nSaved files:")

print(" - model/best_model.pkl")
print("   (trained model)")

print(" - model/scaler.pkl")
print("   (fitted StandardScaler)")

print(" - model/feature_columns.pkl")
print("   (expected feature order)")

print(" - model/model_name.pkl")
print("   (name of the selected model)")


print("\nTraining complete! 🎉")