# 🎓 Student Performance Predictor

A machine learning web application that predicts a student's final performance score based on academic and lifestyle-related factors.

The project combines a **React frontend**, **FastAPI backend**, and **Scikit-learn machine learning models** to create an end-to-end ML prediction system.

---

## 📌 Project Overview

The Student Performance Predictor takes the following information from a user:

- Study hours
- Attendance percentage
- Previous exam score
- Sleep hours
- Assignment completion percentage
- Extracurricular activity participation

The input is processed and passed to a trained machine learning model, which predicts the student's expected final score.

The project demonstrates the complete machine learning workflow:

**Data → Preprocessing → Training → Evaluation → Model Selection → Prediction → API → Web Interface**

---

## ✨ Features

- 📊 Student performance prediction
- 🧹 Missing-value handling
- 🔢 Categorical feature encoding
- ⚖️ Feature scaling using `StandardScaler`
- 🤖 Linear Regression model
- 🌲 Random Forest Regression model
- 📈 Model evaluation using:
  - MAE
  - MSE
  - RMSE
  - R² Score
- 🏆 Automatic selection of the better-performing model
- 💾 Saved trained model using Joblib
- ⚡ FastAPI prediction API
- ⚛️ React + Vite frontend
- 🔗 Frontend-to-backend integration

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Backend

- FastAPI
- Uvicorn

### Frontend

- React
- Vite
- JavaScript
- CSS

### Development Tools

- VS Code
- Git
- GitHub

---

## 📂 Project Structure

```text
student-performance-predictor/
│
├── data/
│   └── student_performance.csv
│
├── model/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── model_name.pkl
│
├── public/
│
├── src/
│   ├── assets/
│   ├── App.jsx
│   ├── App.css
│   ├── index.css
│   └── main.jsx
│
├── generate_dataset.py
├── train_model.py
├── predict.py
├── main.py
├── requirements.txt
├── package.json
├── package-lock.json
├── vite.config.js
└── README.md