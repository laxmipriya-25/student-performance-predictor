import json
import os

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_PATH = "data/student_performance.csv"
MODEL_PATH = "model/from_scratch_model.json"

FEATURE_COLUMNS = [
    "study_hours",
    "attendance_percentage",
    "previous_score",
    "sleep_hours",
    "assignment_completion",
    "extracurricular",
]

TARGET_COLUMN = "final_score"


# ------------------------------------------------------------
# Load and preprocess data
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Handle missing numeric values
numeric_columns = [
    "study_hours",
    "attendance_percentage",
    "previous_score",
    "sleep_hours",
    "assignment_completion",
]

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

# Encode extracurricular
df["extracurricular"] = (
    df["extracurricular"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

# Check for unexpected categorical values
if df["extracurricular"].isnull().any():
    raise ValueError("Unexpected value found in extracurricular column.")

X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
y = df[TARGET_COLUMN].to_numpy(dtype=float)


# ------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------

rng = np.random.default_rng(42)

indices = np.arange(len(X))
rng.shuffle(indices)

X = X[indices]
y = y[indices]

split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# ------------------------------------------------------------
# Feature scaling
# ------------------------------------------------------------

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

# Prevent division by zero
train_std[train_std == 0] = 1.0

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std


# ------------------------------------------------------------
# Linear Regression from scratch
# ------------------------------------------------------------

def predict(X, weights, bias):
    return X @ weights + bias


def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


def train_linear_regression(
    X,
    y,
    learning_rate=0.01,
    iterations=5000
):
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for iteration in range(iterations):

        # Forward pass
        predictions = predict(X, weights, bias)

        # Error
        error = predictions - y

        # Gradients
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Print progress occasionally
        if iteration % 1000 == 0:
            loss = compute_mse(y, predictions)
            print(
                f"Iteration {iteration:4d} | "
                f"MSE: {loss:.4f}"
            )

    return weights, bias


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\n==============================================")
print("FROM-SCRATCH LINEAR REGRESSION")
print("==============================================")

print(f"Dataset shape: {df.shape}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")

print("\nTraining model...\n")

weights, bias = train_linear_regression(
    X_train_scaled,
    y_train,
    learning_rate=0.01,
    iterations=5000
)


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

train_predictions = predict(
    X_train_scaled,
    weights,
    bias
)

test_predictions = predict(
    X_test_scaled,
    weights,
    bias
)

train_mse = compute_mse(y_train, train_predictions)
test_mse = compute_mse(y_test, test_predictions)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_r2 = compute_r2(y_train, train_predictions)
test_r2 = compute_r2(y_test, test_predictions)


print("\n==============================================")
print("MODEL RESULTS")
print("==============================================")

print(f"Training MSE  : {train_mse:.4f}")
print(f"Test MSE      : {test_mse:.4f}")

print(f"Training RMSE : {train_rmse:.4f}")
print(f"Test RMSE     : {test_rmse:.4f}")

print(f"Training R²   : {train_r2:.4f}")
print(f"Test R²       : {test_r2:.4f}")


print("\nLearned coefficients:")
for feature, weight in zip(FEATURE_COLUMNS, weights):
    print(f"{feature:25s}: {weight:.4f}")

print(f"{'intercept':25s}: {bias:.4f}")


# ------------------------------------------------------------
# Save model parameters
# ------------------------------------------------------------

os.makedirs("model", exist_ok=True)

model_data = {
    "model_type": "Linear Regression from Scratch",
    "feature_columns": FEATURE_COLUMNS,
    "weights": weights.tolist(),
    "bias": float(bias),
    "training_mean": train_mean.tolist(),
    "training_std": train_std.tolist(),
    "training_mse": float(train_mse),
    "test_mse": float(test_mse),
    "training_rmse": float(train_rmse),
    "test_rmse": float(test_rmse),
    "training_r2": float(train_r2),
    "test_r2": float(test_r2),
    "learning_rate": 0.01,
    "iterations": 5000,
    "random_seed": 42,
}

with open(MODEL_PATH, "w") as file:
    json.dump(model_data, file, indent=4)


print("\n==============================================")
print("MODEL SAVED")
print("==============================================")
print(f"Saved to: {MODEL_PATH}")