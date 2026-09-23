import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.predict import EmotionPredictor

app = Flask(__name__)
CORS(app)

# Initialize predictor
try:
    predictor = EmotionPredictor()
    print("Emotion Predictor loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model ({e}). Train model first via python src/train.py.")
    predictor = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict_emotion():
    if not predictor:
        return jsonify({"error": "Model not loaded. Please run model training."}), 500
        
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
        
    result = predictor.predict(text)
    return jsonify(result)

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = os.path.join(app.root_path, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Metrics file not found."}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
