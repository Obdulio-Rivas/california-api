"""
app.py — API Flask para predicción de precios de viviendas en California.

Local:
    python app.py

Produccion (gunicorn):
    gunicorn app:app --bind 0.0.0.0:$PORT
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.getenv("MODEL_PATH", "models/california_housing_model.pkl")

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
metrics  = artifact["metrics"]
features = artifact["features"]  # lista de features que espera el modelo

REQUIRED_FIELDS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def validate_input(data) -> str | None:
    """Valida el cuerpo de entrada"""
    if not isinstance(data, dict):
        return "El body debe ser un objeto JSON"
    for field in REQUIRED_FIELDS:
        if field not in data:
            return f"Campo requerido faltante: '{field}'"
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"'{field}' debe ser un valor numerico"
    return None


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024


@app.route("/")
def root():
    return jsonify({
        "service":     "California Housing Price Predictor",
        "version":     "1.0.0",
        "description": "Realiza la prediccion de precios de viviendas en California (cientos de miles USD).",
        "endpoints": {
            "GET  /":        "Informacion del servicio",
            "GET  /health":  "Estado del servicio y metricas del modelo",
            "POST /predict": "Prediccion de precio dado las features de la vivienda",
            "GET  /features": "Lista las features requeridas por el modelo",
        },
    })


@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "model":   "XGBoost + StandardScaler pipeline",
        "metrics": metrics,
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "El body debe ser JSON valido"}), 400

    error = validate_input(data)
    if error:
        return jsonify({"error": error}), 400

    try:
        # Construir DataFrame con solo las features del modelo
        input_df = pd.DataFrame([{field: data[field] for field in REQUIRED_FIELDS}])
        prediction = pipeline.predict(input_df)[0]
        # El modelo predice en escala original (cientos de miles USD)
        predicted_value = float(prediction)
    except Exception as exc:
        app.logger.error("Error en /predict: %s", exc, exc_info=True)
        return jsonify({"error": "Error interno al procesar la prediccion"}), 500

    return jsonify({
        "predicted_value": round(predicted_value, 4),
        "unit": "cientos de miles USD (escala original del dataset)",
    })


@app.route("/features")
def get_features():
    return jsonify({
        "required_features": REQUIRED_FIELDS,
        "description": {
            "MedInc":     "Ingreso mediano del bloque (decenas de miles USD)",
            "HouseAge":   "Edad mediana de las viviendas del bloque (anios)",
            "AveRooms":   "Promedio de habitaciones por vivienda",
            "AveBedrms":  "Promedio de dormitorios por vivienda",
            "Population": "Poblacion del bloque",
            "AveOccup":   "Promedio de ocupantes por vivienda",
            "Latitude":   "Latitud del bloque",
            "Longitude":  "Longitud del bloque",
        },
        "output": {
            "predicted_value": "Valor mediano estimado de la vivienda (cientos de miles USD)",
        },
    })


@app.errorhandler(404)
def not_found(exc):
    return jsonify({"error": "Endpoint no encontrado"}), 404


@app.errorhandler(405)
def method_not_allowed(exc):
    return jsonify({"error": "Metodo HTTP no permitido en este endpoint"}), 405


@app.errorhandler(413)
def request_too_large(exc):
    return jsonify({"error": "Request demasiado grande"}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
