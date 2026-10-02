"""
predict.py

Prediction module for the Student Performance Prediction project.

Supports:
1. Existing sklearn model
2. From-scratch Linear Regression model

The from-scratch model is implemented using NumPy and saved
parameters from from_scratch_regression.py.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Existing sklearn model files
# ------------------------------------------------------------

MODEL_PATH = "model/best_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_COLUMNS_PATH = "model/feature_columns.pkl"
MODEL_NAME_PATH = "model/model_name.pkl"


# ------------------------------------------------------------
# From-scratch model file
# ------------------------------------------------------------

FROM_SCRATCH_MODEL_PATH = "model/from_scratch_model.json"


# ------------------------------------------------------------
# Load existing sklearn model
# ------------------------------------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
model_name = joblib.load(MODEL_NAME_PATH)


# ------------------------------------------------------------
# Load from-scratch model
# ------------------------------------------------------------

with open(FROM_SCRATCH_MODEL_PATH, "r") as file:
    from_scratch_data = json.load(file)


from_scratch_features = from_scratch_data["feature_columns"]
from_scratch_weights = np.array(
    from_scratch_data["weights"],
    dtype=float
)
from_scratch_bias = float(
    from_scratch_data["bias"]
)
from_scratch_mean = np.array(
    from_scratch_data["training_mean"],
    dtype=float
)
from_scratch_std = np.array(
    from_scratch_data["training_std"],
    dtype=float
)


# ------------------------------------------------------------
# From-scratch prediction function
# ------------------------------------------------------------

def predict_from_scratch(input_data: pd.DataFrame) -> float:
    """
    Predict using the manually implemented
    linear regression model.
    """

    X = input_data[from_scratch_features].to_numpy(
        dtype=float
    )

    # Apply the SAME scaling used during training
    X_scaled = (
        X - from_scratch_mean
    ) / from_scratch_std

    # Linear regression:
    # prediction = Xw + b
    prediction = (
        X_scaled @ from_scratch_weights
        + from_scratch_bias
    )

    return float(
        np.clip(prediction[0], 0, 100)
    )


# ------------------------------------------------------------
# Main prediction function
# ------------------------------------------------------------

def predict_score(
    study_hours: float,
    attendance_percentage: float,
    previous_score: float,
    sleep_hours: float,
    assignment_completion: float,
    extracurricular: str,
    model_type: str = "from_scratch"
) -> dict:

    """
    Predict a student's final performance score.

    model_type:
        "from_scratch" -> our manually implemented model
        "sklearn"      -> existing trained sklearn model
    """

    # --------------------------------------------------------
    # Encode categorical input
    # --------------------------------------------------------

    extracurricular_numeric = (
        1
        if str(extracurricular).strip().lower() == "yes"
        else 0
    )


    # --------------------------------------------------------
    # Build input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "study_hours": study_hours,

        "attendance_percentage":
            attendance_percentage,

        "previous_score":
            previous_score,

        "sleep_hours":
            sleep_hours,

        "assignment_completion":
            assignment_completion,

        "extracurricular":
            extracurricular_numeric,

    }])


    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if model_type == "from_scratch":

        prediction = predict_from_scratch(
            input_data
        )

        model_used = (
            "Linear Regression "
            "(From Scratch)"
        )

    elif model_type == "sklearn":

        # Match the feature order expected
        # by the existing sklearn model.

        input_data = input_data[
            feature_columns
        ]

        # Apply sklearn scaler

        input_scaled = scaler.transform(
            input_data
        )

        # Predict

        prediction = model.predict(
            input_scaled
        )[0]

        prediction = float(
            np.clip(prediction, 0, 100)
        )

        model_used = model_name

    else:

        raise ValueError(
            "model_type must be "
            "'from_scratch' or 'sklearn'"
        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "predicted_final_score":
            round(prediction, 2),

        "model_used":
            model_used,

    }


# ------------------------------------------------------------
# Standalone testing
# ------------------------------------------------------------

if __name__ == "__main__":

    example = {

        "study_hours": 6,

        "attendance_percentage": 92,

        "previous_score": 78,

        "sleep_hours": 7.5,

        "assignment_completion": 95,

        "extracurricular": "Yes",

    }


    print(
        "\n=============================="
    )

    print(
        "FROM-SCRATCH MODEL TEST"
    )

    print(
        "=============================="
    )

    result = predict_score(
        **example,
        model_type="from_scratch"
    )

    print(
        "Input:",
        example
    )

    print(
        "Prediction:",
        result
    )


    print(
        "\n=============================="
    )

    print(
        "SKLEARN MODEL TEST"
    )

    print(
        "=============================="
    )

    result_sklearn = predict_score(
        **example,
        model_type="sklearn"
    )

    print(
        "Prediction:",
        result_sklearn
    )