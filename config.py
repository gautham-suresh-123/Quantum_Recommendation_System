"""
Configuration module for Simplified Quantum-Assisted Personalized Recommendation System.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# MovieLens 100K URL & Files
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
USERS_FILE = DATA_DIR / "users.csv"
MODEL_FILE = MODELS_DIR / "vqc_recommender.pkl"

# Global Random Seed
RANDOM_SEED = 42

# 19 Standard MovieLens Genres
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western", "Unknown"
]

# Preference Rating Threshold (Ratings >= 4.0 are preferred)
PREFERENCE_THRESHOLD = 4.0

# Quantum & PCA Hyperparameters
N_QUBITS = 4
PCA_N_COMPONENTS = 4
VQC_LAYERS = 2

# Training Mode Settings
DEMO_MODE = True  # Set True for fast viva training (2-3 seconds), False for full training
DEMO_TRAIN_SAMPLES = 200
FULL_TRAIN_SAMPLES = 600
COBYLA_MAX_ITER = 25
