"""
QuantumFlix - Research-Grade Movie Recommendation System Web Server.

Serves the standalone HTML5, CSS3, and Vanilla JavaScript application,
exposes the Research Mode and Demo Mode recommendation API endpoints,
and serves empirical research metrics dashboard results.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, send_from_directory, request, jsonify
from src.recommender import process_quantum_recommendations
from config import RESULTS_DIR, MODEL_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    """Renders the single-page recommendation application."""
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    """
    Quantum-Assisted Recommendation API Endpoint.
    Supports 'mode' parameter: 'RESEARCH_MODE' (MovieLens dataset) or 'DEMO_MODE'.
    Returns explicit S_Q, S_C, and S_H scores.
    """
    try:
        user_prefs = request.get_json(force=True, silent=True) or {}
        mode = user_prefs.get("mode", "RESEARCH_MODE")
        result = process_quantum_recommendations(user_prefs, mode=mode)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing quantum recommendation: {e}")
        return jsonify({
            "status": "empty",
            "message": "Error processing quantum recommendation. Please check preference input parameters."
        }), 400


@app.route("/api/results", methods=["GET"])
def results_dashboard_endpoint():
    """
    Research Results Dashboard Endpoint.
    Returns generated empirical metrics from results/*.csv or 'NOT EXECUTED' if unavailable.
    """
    metrics_file = RESULTS_DIR / "metrics.csv"
    circuit_file = RESULTS_DIR / "circuit_metrics.csv"
    ablation_file = RESULTS_DIR / "ablation.csv"
    config_file = RESULTS_DIR / "experiment_config.json"

    response = {
        "status": "success",
        "model_artifact_present": MODEL_FILE.exists()
    }

    if config_file.exists():
        with open(config_file, "r") as f:
            response["experiment_config"] = json.load(f)
    else:
        response["experiment_config"] = "NOT EXECUTED"

    if metrics_file.exists():
        response["metrics"] = pd.read_csv(metrics_file).to_dict(orient="records")
    else:
        response["metrics"] = "NOT EXECUTED"

    if circuit_file.exists():
        response["circuit_metrics"] = pd.read_csv(circuit_file).to_dict(orient="records")
    else:
        response["circuit_metrics"] = "NOT EXECUTED"

    if ablation_file.exists():
        response["ablation"] = pd.read_csv(ablation_file).to_dict(orient="records")
    else:
        response["ablation"] = "NOT EXECUTED"

    return jsonify(response)


@app.route("/style.css")
def serve_css():
    if os.path.exists(os.path.join("templates", "style.css")):
        return send_from_directory("templates", "style.css")
    return send_from_directory("static", "style.css")


@app.route("/script.js")
@app.route("/app.js")
def serve_js():
    if os.path.exists(os.path.join("templates", "script.js")):
        return send_from_directory("templates", "script.js")
    return send_from_directory("static", "script.js")


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Launching QuantumFlix Research Server at http://{host}:{port}...")
    app.run(host=host, port=port, debug=True)
