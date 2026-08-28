from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_score


app = FastAPI(
    title="Student Performance Prediction API"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running!"
    }


@app.post("/predict")
def predict(data: dict):
    result = predict_score(
        study_hours=float(data["study_hours"]),
        attendance_percentage=float(data["attendance_percentage"]),
        previous_score=float(data["previous_score"]),
        sleep_hours=float(data["sleep_hours"]),
        assignment_completion=float(data["assignment_completion"]),
        extracurricular=data["extracurricular"],
    )

    return result