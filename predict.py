"""
predict.py
----------
This script loads the trained model + scaler saved by train_model.py, and
uses them to predict a NEW student's final performance score.

It can be used in two ways:

1. As a standalone script (run directly from the command line):
       python predict.py
   It will run a couple of example predictions using sample student data
   defined in the __main__ section at the bottom of this file.

2. As an importable module (this is the part that matters for later
   connecting to a FastAPI backend):
       from predict import predict_score
       result = predict_score(
           study_hours=5,
           attendance_percentage=85,
           previous_score=70,
           sleep_hours=7,
           assignment_completion=90,
           extracurricular="Yes"
       )
   A FastAPI endpoint can simply call predict_score(...) with values coming
   from a React form and return the result as JSON. No changes to this file
   would be needed for that integration.
"""

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the saved model, scaler, and feature column order ONCE when this
# module is imported (so repeated predictions don't reload files every time)
# ---------------------------------------------------------------------------
MODEL_PATH = "model/best_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_COLUMNS_PATH = "model/feature_columns.pkl"
MODEL_NAME_PATH = "model/model_name.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
model_name = joblib.load(MODEL_NAME_PATH)


def predict_score(study_hours: float,
                   attendance_percentage: float,
                   previous_score: float,
                   sleep_hours: float,
                   assignment_completion: float,
                   extracurricular: str) -> dict:
    """
    Predict a student's final performance score.

    Parameters
    ----------
    study_hours : float
        Average hours studied per day (e.g. 4.5)
    attendance_percentage : float
        Class attendance percentage (0-100)
    previous_score : float
        Score obtained in the previous exam (0-100)
    sleep_hours : float
        Average hours of sleep per day (e.g. 7)
    assignment_completion : float
        Percentage of assignments completed (0-100)
    extracurricular : str
        "Yes" or "No" - whether the student takes part in extracurricular activities

    Returns
    -------
    dict
        {
            "predicted_final_score": float,
            "model_used": str
        }
    """

    # 1. Encode the categorical input the SAME way it was encoded during training
    extracurricular_numeric = 1 if str(extracurricular).strip().lower() == "yes" else 0

    # 2. Build a single-row DataFrame with columns in the EXACT same order
    #    that was used during training (very important for correct predictions)
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "assignment_completion": assignment_completion,
        "extracurricular": extracurricular_numeric,
    }])[feature_columns]

    # 3. Scale the input using the SAME scaler fitted during training
    input_scaled = scaler.transform(input_data)

    # 4. Predict
    prediction = model.predict(input_scaled)[0]

    # 5. Clip the prediction to a realistic 0-100 score range
    prediction = float(np.clip(prediction, 0, 100))

    return {
        "predicted_final_score": round(prediction, 2),
        "model_used": model_name,
    }


# ---------------------------------------------------------------------------
# Standalone script usage: run a few example predictions when this file is
# executed directly (not when it's imported by another script/backend)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print(f"Loaded model: {model_name}\n")

    # Example 1: A hard-working student
    example_1 = {
        "study_hours": 6,
        "attendance_percentage": 92,
        "previous_score": 78,
        "sleep_hours": 7.5,
        "assignment_completion": 95,
        "extracurricular": "Yes",
    }

    # Example 2: A student who is struggling a bit
    example_2 = {
        "study_hours": 1.5,
        "attendance_percentage": 55,
        "previous_score": 40,
        "sleep_hours": 5,
        "assignment_completion": 45,
        "extracurricular": "No",
    }

    for i, example in enumerate([example_1, example_2], start=1):
        result = predict_score(**example)
        print(f"Example {i} input: {example}")
        print(f"Predicted final score: {result['predicted_final_score']}")
        print(f"Model used: {result['model_used']}\n")

    # You can also try entering your own values here:
    custom_prediction = predict_score(
        study_hours=4,
        attendance_percentage=80,
        previous_score=65,
        sleep_hours=6.5,
        assignment_completion=70,
        extracurricular="No"
    )
    print("Custom input prediction:", custom_prediction)
