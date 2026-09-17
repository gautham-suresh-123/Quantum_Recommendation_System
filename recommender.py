"""
Recommender Pipeline Module (Root Proxy Re-export).
All core implementation details reside in `src.recommender`.
"""

from src.recommender import (
    load_model_artifact,
    encode_genre_vector,
    process_quantum_recommendations,
    DEMO_BENCHMARK_MOVIES
)
