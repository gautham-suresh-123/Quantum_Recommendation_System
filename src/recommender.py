"""
Recommendation Pipeline Module (Research Grade).

Ingests user preference parameters, applies pre-fitted Scaler and PCA models from artifact,
evaluates 4-Qubit VQC measurement probability S_Q, computes Classical Content Similarity S_C,
and returns un-clamped Hybrid Recommendations S_H = alpha * S_Q + (1 - alpha) * S_C.

Supports explicit Research Mode (MovieLens dataset catalog) and Demo Mode.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    MODEL_FILE,
    N_QUBITS,
    VQC_LAYERS,
    ALL_GENRES,
    HYBRID_ALPHA,
    MOVIES_FILE,
    RATINGS_FILE
)
from src.preprocessing import (
    encode_genres,
    build_interaction_features,
    transform_preprocessing_pipeline,
    download_and_load_data
)
from src.quantum_model import VariationalQuantumCircuit

# UI Demo Set (Solely for fast frontend UI demonstration mode)
DEMO_BENCHMARK_MOVIES = [
    { "id": 1, "title": "Interstellar", "year": 2014, "genres": ["Sci-Fi", "Drama"], "language": "English", "rating": 8.7, "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival." },
    { "id": 2, "title": "Inception", "year": 2010, "genres": ["Sci-Fi", "Action", "Thriller"], "language": "English", "rating": 8.8, "description": "A thief who steals corporate secrets through dream-sharing technology." },
    { "id": 3, "title": "Dangal", "year": 2016, "genres": ["Drama", "Action"], "language": "Hindi", "rating": 8.3, "description": "Former wrestler Mahavir Singh Phogat trains his daughters to win gold." },
    { "id": 4, "title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"], "language": "English", "rating": 8.7, "description": "A computer hacker learns from mysterious rebels about the true nature of his reality." },
    { "id": 5, "title": "Toy Story", "year": 1995, "genres": ["Animation", "Children's", "Comedy"], "language": "English", "rating": 8.3, "description": "A cowboy doll is profoundly threatened when a new spaceman figure arrives." }
]


def load_model_artifact(model_path: str = str(MODEL_FILE)) -> Optional[Dict[str, Any]]:
    """Loads saved model artifact containing theta weights and pre-fitted transformers."""
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def encode_genre_vector(genres_list: List[str]) -> np.ndarray:
    """Multi-hot encodes genres list into 19D vector."""
    vec = np.zeros(len(ALL_GENRES), dtype=float)
    for g in genres_list:
        if g in ALL_GENRES:
            idx = ALL_GENRES.index(g)
            vec[idx] = 1.0
    return vec


def process_quantum_recommendations(
    user_prefs: Dict[str, Any],
    mode: str = "RESEARCH_MODE"
) -> Dict[str, Any]:
    """
    Executes the 5-Step Quantum Processing Protocol:
    1. Construct 19D User Preference Vector and candidate Movie Profile.
    2. Build 76D Canonical Interaction Feature Vector.
    3. Apply PRE-FITTED Scaler & PCA to reduce 76D -> 4D mapped to [0, pi].
    4. Execute 4-Qubit VQC to obtain un-clamped quantum probability S_Q = P(|1>).
    5. Compute Classical Cosine Similarity S_C and output Hybrid Score S_H = alpha * S_Q + (1 - alpha) * S_C.
    """
    selected_genres = user_prefs.get("genres", [])
    if isinstance(selected_genres, str):
        selected_genres = [selected_genres]
    selected_genres_set = set(selected_genres)

    top_k = int(user_prefs.get("rec_count", 5))
    alpha = float(user_prefs.get("alpha", HYBRID_ALPHA))

    if mode == "RESEARCH_MODE" and MOVIES_FILE.exists():
        movies_df, _ = download_and_load_data()
        genre_features_df = encode_genres(movies_df)
        candidates = []
        for idx, row in movies_df.iterrows():
            m_genres = str(row["genres"]).split("|")
            candidates.append({
                "id": int(row["movieId"]),
                "title": row["title"],
                "genres": m_genres,
                "genres_str": str(row["genres"])
            })
    else:
        candidates = DEMO_BENCHMARK_MOVIES
        genre_features_df = pd.DataFrame([encode_genre_vector(m["genres"]) for m in candidates], index=[m["id"] for m in candidates], columns=ALL_GENRES)

    if selected_genres_set:
        candidates = [m for m in candidates if any(g in selected_genres_set for g in m["genres"])]

    if not candidates:
        return {
            "status": "empty",
            "message": "No candidate movies match the selected genre criteria."
        }

    user_vec = encode_genre_vector(selected_genres)

    artifact = load_model_artifact()
    if artifact:
        theta_params = artifact["optimal_theta"]
        scaler = artifact["scaler"]
        pca_model = artifact["pca_model"]
        angle_scaler = artifact["angle_scaler"]
        vqc = VariationalQuantumCircuit(n_qubits=artifact["n_qubits"], n_layers=artifact["n_layers"])
    else:
        vqc = VariationalQuantumCircuit(n_qubits=N_QUBITS, n_layers=VQC_LAYERS)
        np.random.seed(42)
        theta_params = np.random.uniform(0, 2 * np.pi, vqc.num_params)
        from src.preprocessing import fit_preprocessing_pipeline
        dummy_raw = np.random.rand(10, 76)
        _, pca_model, scaler, angle_scaler = fit_preprocessing_pipeline(dummy_raw, n_components=N_QUBITS)

    scored_movies = []
    for m in candidates:
        mid = m["id"]
        if mid in genre_features_df.index:
            m_prof = genre_features_df.loc[mid].values.astype(float)
        else:
            m_prof = encode_genre_vector(m["genres"])

        raw_feat = build_interaction_features(user_vec, m_prof).reshape(1, -1)
        q_feat = transform_preprocessing_pipeline(raw_feat, scaler, pca_model, angle_scaler)[0]
        s_q = float(vqc.evaluate_preference(q_feat, theta_params))

        u_norm = np.linalg.norm(user_vec)
        m_norm = np.linalg.norm(m_prof)
        if u_norm > 0 and m_norm > 0:
            s_c = float(np.dot(user_vec, m_prof) / (u_norm * m_norm))
        else:
            s_c = 0.5

        s_h = float(alpha * s_q + (1.0 - alpha) * s_c)

        scored_movies.append({
            "id": mid,
            "title": m["title"],
            "genres": m["genres"],
            "quantum_score_sq": round(s_q, 4),
            "classical_score_sc": round(s_c, 4),
            "hybrid_score_sh": round(s_h, 4)
        })

    scored_movies.sort(key=lambda x: x["hybrid_score_sh"], reverse=True)
    top_recs = scored_movies[:top_k]

    return {
        "status": "success",
        "mode": mode,
        "model_artifact_loaded": artifact is not None,
        "hybrid_alpha": alpha,
        "top_match": top_recs[0] if top_recs else None,
        "recommendations": top_recs
    }
