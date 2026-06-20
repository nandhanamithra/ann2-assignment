"""
Obesity Level Predictor — Flask API
------------------------------------
Loads the artifacts produced by train_model.py:
  - obesity_model.keras   (preferred)  OR  obs_model.pkl (legacy joblib dump)
  - obs_scaler.pkl
  - obs_encoder.pkl
  - obs_target_encoder.pkl

Endpoints
  GET  /            -> serves the UI
  GET  /api/meta     -> field metadata the UI uses to build the form
  POST /api/predict  -> { "Gender": "Male", "Age": 23, ... }  ->  JSON result
  GET  /api/health   -> readiness check
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

ARTIFACT_DIR = Path(__file__).resolve().parent

# Fallback column order, only used if the saved scaler doesn't carry
# feature_names_in_ (older scikit-learn). When the scaler was fit on a
# pandas DataFrame -- as in train_model.py -- this is set automatically,
# so the API never has to guess the training column order.
FALLBACK_FEATURE_ORDER = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE",
    "CALC", "MTRANS",
]

# Reasonable slider bounds for the numeric fields in this dataset. Adjust
# if your CSV's actual min/max differ.
NUMERIC_RANGES = {
    "Age": {"min": 10, "max": 80, "step": 1, "default": 25},
    "Height": {"min": 1.40, "max": 2.10, "step": 0.01, "default": 1.70},
    "Weight": {"min": 30, "max": 200, "step": 0.5, "default": 70},
    "FCVC": {"min": 1, "max": 3, "step": 0.1, "default": 2},
    "NCP": {"min": 1, "max": 4, "step": 0.1, "default": 3},
    "CH2O": {"min": 1, "max": 3, "step": 0.1, "default": 2},
    "FAF": {"min": 0, "max": 3, "step": 0.1, "default": 1},
    "TUE": {"min": 0, "max": 2, "step": 0.1, "default": 1},
}

app = Flask(__name__)


def load_artifacts():
    """Load model + preprocessing artifacts. Returns Nones for anything
    that isn't found so the app can still boot and report status."""
    model = None

    # Preferred: native Keras format saved with model.save(...)
    for keras_name in ("obesity_model.keras", "obs_model.keras", "obs_model.h5"):
        candidate = ARTIFACT_DIR / keras_name
        if candidate.exists():
            try:
                from tensorflow.keras.models import load_model
                model = load_model(candidate)
                print(f"Loaded Keras model from {candidate.name}")
            except Exception as exc:  # pragma: no cover
                print(f"Failed to load {candidate.name}: {exc}")
            break

    # Legacy fallback: the model was joblib-dumped directly (works only on
    # some TensorFlow/Keras versions -- model.save() is the supported path).
    if model is None:
        legacy_path = ARTIFACT_DIR / "obs_model.pkl"
        if legacy_path.exists():
            try:
                model = joblib.load(legacy_path)
                print("Loaded legacy joblib-pickled model (obs_model.pkl)")
            except Exception as exc:  # pragma: no cover
                print(f"Failed to load obs_model.pkl via joblib: {exc}")

    def _load_pkl(name):
        path = ARTIFACT_DIR / name
        if path.exists():
            return joblib.load(path)
        return None

    scaler = _load_pkl("obs_scaler.pkl")
    encoders = _load_pkl("obs_encoder.pkl") or {}
    target_encoder = _load_pkl("obs_target_encoder.pkl")

    return model, scaler, encoders, target_encoder


model, scaler, encoders, target_encoder = load_artifacts()


def get_feature_order():
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return FALLBACK_FEATURE_ORDER


def predict_one(payload: dict):
    """Validate + encode + scale + predict a single record.
    Returns (result_dict, error_dict). Exactly one of the two is None."""
    feature_order = get_feature_order()

    row, missing, invalid = [], [], []
    for col in feature_order:
        if col not in payload or payload[col] in (None, ""):
            missing.append(col)
            continue

        value = payload[col]
        if col in encoders:
            le = encoders[col]
            try:
                row.append(le.transform([str(value)])[0])
            except ValueError:
                allowed = [str(c) for c in le.classes_]
                invalid.append({"field": col, "reason": f"must be one of {allowed}"})
        else:
            try:
                row.append(float(value))
            except (TypeError, ValueError):
                invalid.append({"field": col, "reason": "must be a number"})

    if missing or invalid:
        return None, {"missing_fields": missing, "invalid_fields": invalid}

    # One-row DataFrame (not a bare array) keeps column names aligned with
    # what the scaler was fit on, avoiding sklearn's "X does not have valid
    # feature names" warning.
    x_df = pd.DataFrame([row], columns=feature_order)
    x_scaled = scaler.transform(x_df)

    raw_pred = model.predict(x_scaled, verbose=0) if hasattr(model, "predict") else None
    probs = np.asarray(raw_pred)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = target_encoder.inverse_transform([pred_idx])[0]

    classes = list(target_encoder.classes_)
    probability_map = {cls: round(float(p), 6) for cls, p in zip(classes, probs)}

    return {
        "prediction": str(pred_label),
        "confidence": round(float(probs[pred_idx]), 6),
        "probabilities": probability_map,
    }, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify(
        status="ok" if all([model, scaler, encoders, target_encoder]) else "missing_artifacts",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        encoders_loaded=bool(encoders),
        target_encoder_loaded=target_encoder is not None,
    )


@app.route("/api/meta")
def meta():
    feature_order = get_feature_order()
    categorical_fields = {col: [str(c) for c in le.classes_] for col, le in encoders.items()}
    numeric_fields = {
        col: NUMERIC_RANGES.get(col, {"min": 0, "max": 10, "step": 1, "default": 5})
        for col in feature_order
        if col not in categorical_fields
    }
    target_classes = list(target_encoder.classes_) if target_encoder is not None else []

    return jsonify(
        feature_order=feature_order,
        categorical_fields=categorical_fields,
        numeric_fields=numeric_fields,
        target_classes=target_classes,
        ready=all([model, scaler, encoders, target_encoder]),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if not all([model, scaler, encoders, target_encoder]):
        return jsonify(success=False, error="Model artifacts are not loaded on the server."), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(success=False, error="Request body must be a JSON object."), 400

    result, error = predict_one(payload)
    if error:
        return jsonify(success=False, error="Validation failed", **error), 400

    return jsonify(success=True, **result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
