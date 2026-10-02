import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv("data/student_performance.csv")


# Handle missing numeric values
numeric_columns = [
    "study_hours",
    "attendance_percentage",
    "previous_score",
    "sleep_hours",
    "assignment_completion",
]

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


# Encode extracurricular
df["extracurricular"] = (
    df["extracurricular"]
    .map({"Yes": 1, "No": 0})
    .fillna(0)
)


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

features = [
    "study_hours",
    "attendance_percentage",
    "previous_score",
    "sleep_hours",
    "assignment_completion",
    "extracurricular",
]

X = df[features]
y = df["final_score"]


# ---------------------------------------------------------
# Same train/test split
# ---------------------------------------------------------

rng = np.random.default_rng(42)

indices = np.arange(len(X))
rng.shuffle(indices)

split_index = int(0.8 * len(X))

train_idx = indices[:split_index]
test_idx = indices[split_index:]


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# ---------------------------------------------------------
# FROM-SCRATCH MODEL
# ---------------------------------------------------------

with open("model/from_scratch_model.json", "r") as f:
    scratch = json.load(f)


weights = np.array(
    scratch["weights"],
    dtype=float
)

bias = float(
    scratch["bias"]
)

training_mean = np.array(
    scratch["training_mean"],
    dtype=float
)

training_std = np.array(
    scratch["training_std"],
    dtype=float
)


X_test_scratch = X_test.to_numpy(
    dtype=float
)

X_test_scratch_scaled = (
    X_test_scratch - training_mean
) / training_std


scratch_predictions = (
    X_test_scratch_scaled @ weights
    + bias
)


# ---------------------------------------------------------
# SKLEARN MODEL
# ---------------------------------------------------------

model = joblib.load(
    "model/best_model.pkl"
)

scaler = joblib.load(
    "model/scaler.pkl"
)

# Keep pandas DataFrame so feature names match
X_test_sklearn_scaled = scaler.transform(
    X_test
)

sklearn_predictions = model.predict(
    X_test_sklearn_scaled
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

scratch_mse = mean_squared_error(
    y_test,
    scratch_predictions
)

scratch_rmse = np.sqrt(
    scratch_mse
)

scratch_r2 = r2_score(
    y_test,
    scratch_predictions
)


sklearn_mse = mean_squared_error(
    y_test,
    sklearn_predictions
)

sklearn_rmse = np.sqrt(
    sklearn_mse
)

sklearn_r2 = r2_score(
    y_test,
    sklearn_predictions
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("==============================================")
print("MODEL COMPARISON")
print("==============================================")

print(
    f"{'Metric':<15}"
    f"{'From Scratch':>18}"
    f"{'Sklearn':>15}"
)

print("-" * 48)

print(
    f"{'MSE':<15}"
    f"{scratch_mse:>18.4f}"
    f"{sklearn_mse:>15.4f}"
)

print(
    f"{'RMSE':<15}"
    f"{scratch_rmse:>18.4f}"
    f"{sklearn_rmse:>15.4f}"
)

print(
    f"{'R²':<15}"
    f"{scratch_r2:>18.4f}"
    f"{sklearn_r2:>15.4f}"
)

print("==============================================")