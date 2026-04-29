from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = []

        features.append(float(request.form["Time"]))
        for i in range(1,29):
            features.append(float(request.form[f"V{i}"]))
        features.append(float(request.form["Amount"]))

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        if prediction[0] == 1:
            result = "⚠ High Risk: Fraudulent Transaction Detected"
        else:
            result = "✅ Low Risk: Transaction Appears Safe"

        return render_template(
            "index.html",
            prediction_text=result,
            fraud_probability=probability
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)