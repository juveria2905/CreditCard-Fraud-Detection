from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("fraud_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = []

        features.append(float(request.form["Time"]))
        for i in range(1, 29):
            features.append(float(request.form[f"V{i}"]))
        features.append(float(request.form["Amount"]))

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            result = "⚠ Fraud Detected"
            color = "danger"
        else:
            result = "✅ Safe Transaction"
            color = "success"

        return render_template(
            "index.html",
            prediction_text=result,
            probability=round(probability * 100, 2),
            alert_color=color
        )

    except:
        return render_template("index.html", prediction_text="Error in input")

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)
