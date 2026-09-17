"""
Configuration Module for Research-Grade Hybrid Quantum-Classical Movie Recommendation System.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PAPER_DIR = BASE_DIR / "paper"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PAPER_DIR.mkdir(exist_ok=True)
(PAPER_DIR / "tables").mkdir(exist_ok=True)
(PAPER_DIR / "figures").mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Dataset URLs & Files
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
USERS_FILE = DATA_DIR / "users.csv"
MODEL_FILE = MODELS_DIR / "vqc_recommender.pkl"

# Global Random Seeds for Reproducibility
RANDOM_SEED = 42
EXPERIMENTAL_SEEDS = [42, 52, 62, 72, 82]

# 19 Standard MovieLens Genres
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western", "Unknown"
]

# Preference Rating Threshold (Ratings >= 4.0 are preferred / positive interactions)
PREFERENCE_THRESHOLD = 4.0

# Feature & Dimensionality Reduction Parameters
CANONICAL_FEATURE_DIM = 76  # 19 (alignment) + 19 (abs diff) + 19 (user) + 19 (movie)
N_QUBITS = 4
PCA_N_COMPONENTS = 4
VQC_LAYERS = 2

# Hybrid Scoring Weight (S_H = alpha * S_Q + (1 - alpha) * S_C)
HYBRID_ALPHA = 0.5

# Recommender Evaluation Top-K Settings
EVAL_TOP_K_LIST = [5, 10]

# Optimizer Settings
COBYLA_MAX_ITER = 50
