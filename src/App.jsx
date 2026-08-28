import { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    study_hours: "",
    attendance_percentage: "",
    previous_score: "",
    sleep_hours: "",
    assignment_completion: "",
    extracurricular: "Yes",
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
  e.preventDefault();

  try {
    const response = await fetch("http://127.0.0.1:8001/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      throw new Error("Prediction request failed");
    }

    const result = await response.json();

    setPrediction(
      `Predicted Final Score: ${result.predicted_final_score}`
    );
  } catch (error) {
    console.error(error);
    setPrediction("Could not connect to the prediction server.");
  }
};
  return (
    <div className="app">
      <div className="container">
        <h1>Student Performance Predictor</h1>

        <p className="subtitle">
          Predict a student's final score using academic and lifestyle factors.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Study Hours per Day</label>
            <input
              type="number"
              name="study_hours"
              value={formData.study_hours}
              onChange={handleChange}
              placeholder="e.g. 5"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label>Attendance Percentage</label>
            <input
              type="number"
              name="attendance_percentage"
              value={formData.attendance_percentage}
              onChange={handleChange}
              placeholder="e.g. 85"
              min="0"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label>Previous Score</label>
            <input
              type="number"
              name="previous_score"
              value={formData.previous_score}
              onChange={handleChange}
              placeholder="e.g. 72"
              min="0"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label>Sleep Hours per Day</label>
            <input
              type="number"
              name="sleep_hours"
              value={formData.sleep_hours}
              onChange={handleChange}
              placeholder="e.g. 7"
              min="0"
              max="24"
              step="0.1"
              required
            />
          </div>

          <div className="form-group">
            <label>Assignment Completion (%)</label>
            <input
              type="number"
              name="assignment_completion"
              value={formData.assignment_completion}
              onChange={handleChange}
              placeholder="e.g. 90"
              min="0"
              max="100"
              required
            />
          </div>

          <div className="form-group">
            <label>Extracurricular Activities</label>

            <select
              name="extracurricular"
              value={formData.extracurricular}
              onChange={handleChange}
            >
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>

          <button type="submit">
            Predict Final Score
          </button>
        </form>

        {prediction && (
          <div className="result">
            <h2>Prediction</h2>
            <p>{prediction}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;