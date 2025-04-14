import os
import xgboost as xgb
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.serving import run_simple
from flask import render_template


# Setup paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/loan_default_tuned/loan_default_tuned_model.json")
FEATURES_PATH = os.path.join(BASE_DIR, "../models/loan_default_v1/feature_names.txt")
SOCKET_PATH = "/run/loan-api.sock"  # This should match the .service config

# Load XGBoost model
model = xgb.Booster()
model.load_model(MODEL_PATH)

# Load feature names
with open(FEATURES_PATH, "r") as f:
    FEATURES = f.read().strip().split(",")

# Create Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Empty or invalid JSON input"}), 400

        df = pd.DataFrame([input_data], columns=FEATURES)
        dmatrix = xgb.DMatrix(df)
        pred = model.predict(dmatrix)[0]

        return jsonify({
            "default_probability": round(float(pred), 4),
            "prediction": "Default" if pred > 0.5 else "Fully Paid"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    run_simple(
        hostname="unix://" + SOCKET_PATH,
        port=0,
        application=app,
        use_reloader=False,
        use_debugger=False,
        threaded=True
    )
