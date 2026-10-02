# Student Performance Predictor

A full-stack machine learning application that predicts a student's final academic score using academic and lifestyle-related factors.

The project combines a React frontend, FastAPI backend, and two Linear Regression implementations:
- Linear Regression implemented from scratch using NumPy
- Linear Regression implemented using Scikit-learn

The two models are evaluated using the same test dataset and compared using MSE, RMSE, and R².

---

## 🚀 Features

- Predict student final scores through a web interface
- React-based frontend
- FastAPI REST API backend
- Linear Regression implemented from scratch
- Gradient Descent optimization using NumPy
- Scikit-learn Linear Regression implementation
- Model performance comparison
- MSE, RMSE, and R² evaluation
- Saved model parameters for prediction
- CORS-enabled API for frontend integration

---

## 🧠 Machine Learning Approach

The model uses six input features:

| Feature | Description |
|---|---|
| Study Hours | Average study hours per day |
| Attendance Percentage | Student attendance percentage |
| Previous Score | Previous academic score |
| Sleep Hours | Average sleep hours per day |
| Assignment Completion | Percentage of assignments completed |
| Extracurricular | Whether the student participates in extracurricular activities |

The target variable is:

**Final Score**

---

## 🔬 Linear Regression From Scratch

Instead of relying entirely on a machine learning library, Linear Regression was implemented manually using NumPy.

The implementation includes:

- Data preprocessing
- Train/test splitting
- Feature standardization
- Weight and bias initialization
- Prediction
- Mean Squared Error calculation
- Gradient Descent
- R² evaluation
- Model parameter saving

The trained parameters are stored in:

```text
model/from_scratch_model.json

Model Comparison

The models were evaluated on a test split containing 100 samples.
| Metric | From Scratch | Scikit-learn |
| ------ | -----------: | -----------: |
| MSE    |      30.4688 |      30.2358 |
| RMSE   |       5.5199 |       5.4987 |
| R²     |       0.7392 |       0.7412 |

Project Structure
student-performance-predictor/
│
├── data/
│   └── student_performance.csv
│
├── model/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── model_name.pkl
│   └── from_scratch_model.json
│
├── public/
│
├── src/
│   ├── App.jsx
│   └── App.css
│
├── from_scratch_regression.py
├── model_comparison.py
├── train_model.py
├── predict.py
├── generate_dataset.py
├── main.py
├── requirements.txt
└── README.md

⚙️ Technologies Used
Frontend
React
JavaScript
CSS

Backend
Python
FastAPI
Uvicorn

Machine Learning
NumPy
Pandas
Scikit-learn
Joblib