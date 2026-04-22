import pickle
import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import encode_single, load_encoders, FEATURES

app = Flask(__name__)
CORS(app)

BASE = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH    = os.path.join(BASE, "models", "model.pkl")
ENCODERS_PATH = os.path.join(BASE, "models", "encoders.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

encoders = load_encoders(ENCODERS_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()

    required = ["total_funding", "num_rounds", "avg_round_size", "max_round_size",
                "days_to_first_funding", "funding_duration_days", "founded_year",
                "country_code", "category_code"]

    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        data = encode_single(dict(body), encoders)
        row = np.array([[data.get(f, 0) for f in FEATURES]])
        prob     = float(model.predict_proba(row)[0][1])
        survived = int(model.predict(row)[0])

        return jsonify({
            "survived":    survived,
            "probability": round(prob, 4),
            "label":       "Likely to Survive" if survived == 1 else "Likely to Fail",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
