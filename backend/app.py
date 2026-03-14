"""
Flask API — Blood Group Prediction from Fingerprint
====================================================
Endpoints:
    POST /predict   — accepts image file → returns blood group + confidence
    GET  /health    — health check

Run:
    python app.py
"""

import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH       = os.environ.get("MODEL_PATH", "../model_output/blood_group_model.keras")
CLASS_INDEX_PATH = os.environ.get("CLASS_INDEX_PATH", "../model_output/class_indices.json")
IMG_SIZE         = (224, 224)
ALLOWED_EXT      = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

app = Flask(__name__)
CORS(app)  # allow requests from frontend

# ── Load model once at startup ───────────────────────────────────────────────
print("⏳  Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅  Model loaded")

with open(CLASS_INDEX_PATH) as f:
    raw_indices = json.load(f)
# Invert: index → class_name
idx_to_class = {v: k for k, v in raw_indices.items()}
print(f"✅  Classes: {idx_to_class}")

# ── Blood group descriptions ──────────────────────────────────────────────────
BG_INFO = {
    "A+":  {"donate": "A+, AB+",       "receive": "A+, A-, O+, O-", "population": "~30%"},
    "A-":  {"donate": "A+, A-, AB+, AB-", "receive": "A-, O-",      "population": "~8%"},
    "B+":  {"donate": "B+, AB+",       "receive": "B+, B-, O+, O-", "population": "~9%"},
    "B-":  {"donate": "B+, B-, AB+, AB-", "receive": "B-, O-",      "population": "~2%"},
    "AB+": {"donate": "AB+",           "receive": "All types",       "population": "~3%"},
    "AB-": {"donate": "AB+, AB-",      "receive": "AB-, A-, B-, O-", "population": "~1%"},
    "O+":  {"donate": "A+, B+, O+, AB+", "receive": "O+, O-",       "population": "~38%"},
    "O-":  {"donate": "All types",     "receive": "O- only",         "population": "~7%"},
}


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Use: {ALLOWED_EXT}"}), 400

    try:
        img_bytes = file.read()
        tensor    = preprocess(img_bytes)
        preds     = model.predict(tensor, verbose=0)[0]

        top_idx   = int(np.argmax(preds))
        top_label = idx_to_class[top_idx]
        top_conf  = float(preds[top_idx]) * 100

        # All class probabilities
        all_probs = [
            {"blood_group": idx_to_class[i], "confidence": round(float(p) * 100, 2)}
            for i, p in enumerate(preds)
        ]
        all_probs.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "blood_group": top_label,
            "confidence":  round(top_conf, 2),
            "all_probabilities": all_probs,
            "info": BG_INFO.get(top_label, {}),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)