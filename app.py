"""
QuantumFlix - Movie Recommendation System Web Server
Serves the standalone HTML5, CSS3, and Vanilla JavaScript application,
and exposes the Quantum-Assisted Recommendation AI Engine endpoint.
"""

import os
import logging
from flask import Flask, render_template, send_from_directory, request, jsonify
from recommender import process_quantum_recommendations

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
    Ingests user preference parameters, processes via 4-Qubit VQC pipeline,
    and returns JSON matching the frontend specification.
    """
    try:
        user_prefs = request.get_json(force=True, silent=True) or {}
        result = process_quantum_recommendations(user_prefs)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing quantum recommendation: {e}")
        return jsonify({
            "status": "empty",
            "message": "Error processing quantum recommendation. Please check preference input parameters."
        }), 400

@app.route("/style.css")
def serve_css():
    """Fallback route for style.css."""
    if os.path.exists(os.path.join("templates", "style.css")):
        return send_from_directory("templates", "style.css")
    return send_from_directory("static", "style.css")

@app.route("/script.js")
@app.route("/app.js")
def serve_js():
    """Fallback route for script.js / app.js."""
    if os.path.exists(os.path.join("templates", "script.js")):
        return send_from_directory("templates", "script.js")
    return send_from_directory("static", "script.js")

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Launching QuantumFlix Server at http://{host}:{port}...")
    app.run(host=host, port=port, debug=True)
