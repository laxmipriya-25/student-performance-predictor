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
      const response = await fetch("http://127.0.0.1:8000/predict", {
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

      setPrediction({
        score: result.predicted_final_score,
        model: result.model_used,
      });
    } catch (error) {
      console.error(error);
      setPrediction({
        error: "Could not connect to the prediction server.",
      });
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

            {prediction.error ? (
              <p>{prediction.error}</p>
            ) : (
              <>
                <p>
                  Predicted Final Score:{" "}
                  <strong>{prediction.score}</strong>
                </p>

                <p>
                  Model Used:{" "}
                  <strong>{prediction.model}</strong>
                </p>
              </>
            )}
          </div>
        )}

        <div className="comparison">
          <h2>Model Comparison</h2>

          <p className="comparison-subtitle">
            Performance on the test dataset (100 samples)
          </p>

          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>From Scratch</th>
                <th>Scikit-learn</th>
              </tr>
            </thead>

            <tbody>
              <tr>
                <td>MSE</td>
                <td>30.4688</td>
                <td>30.2358</td>
              </tr>

              <tr>
                <td>RMSE</td>
                <td>5.5199</td>
                <td>5.4987</td>
              </tr>

              <tr>
                <td>R²</td>
                <td>0.7392</td>
                <td>0.7412</td>
              </tr>
            </tbody>
          </table>

          <p className="comparison-note">
            The from-scratch implementation achieves performance very close to
            the Scikit-learn Linear Regression model on this test split.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;