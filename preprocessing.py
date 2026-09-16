"""
Data Preprocessing & Feature Pipeline Module.

Handles MovieLens 100K data loading, multi-hot genre encoding, user preference profile building,
interaction vector construction, and PCA dimensionality reduction.
"""

import os
import io
import zipfile
import logging
import requests
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_DIR,
    MOVIES_FILE,
    RATINGS_FILE,
    USERS_FILE,
    MOVIELENS_100K_URL,
    ALL_GENRES,
    PCA_N_COMPONENTS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads MovieLens 100K dataset automatically if missing and loads cleaned DataFrames.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not (MOVIES_FILE.exists() and RATINGS_FILE.exists()):
        logger.info(f"Downloading MovieLens 100K dataset from {MOVIELENS_100K_URL}...")
        try:
            resp = requests.get(MOVIELENS_100K_URL, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                # Parse movies (u.item)
                item_cols = ["movieId", "title", "release_date", "video_release_date", "IMDb_URL"] + ALL_GENRES
                with z.open("ml-100k/u.item") as f:
                    item_df = pd.read_csv(f, sep="|", encoding="latin-1", names=item_cols)
                
                def get_genre_str(row):
                    active = [g for g in ALL_GENRES if row[g] == 1]
                    return "|".join(active) if active else "Unknown"
                
                item_df["genres"] = item_df.apply(get_genre_str, axis=1)
                item_df[["movieId", "title", "genres"]].to_csv(MOVIES_FILE, index=False)

                # Parse ratings (u.data)
                with z.open("ml-100k/u.data") as f:
                    ratings_df = pd.read_csv(f, sep="\t", names=["userId", "movieId", "rating", "timestamp"])
                ratings_df.to_csv(RATINGS_FILE, index=False)
                logger.info("MovieLens dataset downloaded and saved successfully.")
        except Exception as e:
            logger.warning(f"Network download failed ({e}). Generating fallback data.")
            _generate_fallback_data()

    movies_df = pd.read_csv(MOVIES_FILE)
    ratings_df = pd.read_csv(RATINGS_FILE)
    
    # Cleaning: remove duplicates & handle missing
    movies_df.drop_duplicates(subset=["movieId"], inplace=True)
    movies_df["genres"] = movies_df["genres"].fillna("Unknown")
    
    ratings_df.dropna(subset=["userId", "movieId", "rating"], inplace=True)
    ratings_df.drop_duplicates(subset=["userId", "movieId"], inplace=True)
    
    return movies_df, ratings_df


def _generate_fallback_data():
    """Generates small fallback dataset if offline."""
    movies = [
        {"movieId": 1, "title": "Toy Story (1995)", "genres": "Animation|Children's|Comedy"},
        {"movieId": 2, "title": "GoldenEye (1995)", "genres": "Action|Adventure|Thriller"},
        {"movieId": 3, "title": "Four Rooms (1995)", "genres": "Thriller"},
        {"movieId": 4, "title": "Get Shorty (1995)", "genres": "Action|Comedy|Drama"},
        {"movieId": 5, "title": "Twelve Monkeys (1995)", "genres": "Drama|Sci-Fi"}
    ]
    pd.DataFrame(movies).to_csv(MOVIES_FILE, index=False)
    ratings = []
    for u in range(1, 20):
        for m in range(1, 6):
            ratings.append({"userId": u, "movieId": m, "rating": float(np.random.choice([1, 2, 3, 4, 5])), "timestamp": 881250949})
    pd.DataFrame(ratings).to_csv(RATINGS_FILE, index=False)


def encode_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-hot encodes pipe-separated movie genres into 19-dimensional binary DataFrame indexed by movieId.
    """
    rows = []
    for idx, row in movies_df.iterrows():
        mid = row["movieId"]
        g_split = set(g.strip() for g in str(row["genres"]).split("|"))
        vec = {g: 1 if g in g_split else 0 for g in ALL_GENRES}
        vec["movieId"] = mid
        rows.append(vec)
    df = pd.DataFrame(rows).set_index("movieId")
    return df


def build_user_profile(user_id: int, ratings_df: pd.DataFrame, genre_features_df: pd.DataFrame) -> np.ndarray:
    """
    Calculates normalized user genre preference vector P_u in [0, 1]^19 weighted by rating history.
    """
    u_ratings = ratings_df[ratings_df["userId"] == user_id]
    genre_cols = [g for g in ALL_GENRES if g in genre_features_df.columns]
    
    if len(u_ratings) == 0:
        return np.full(len(genre_cols), 0.5, dtype=float)

    user_mids = u_ratings["movieId"].values
    rated_genres = genre_features_df.loc[genre_features_df.index.isin(user_mids), genre_cols]
    
    r_map = u_ratings.set_index("movieId")["rating"].to_dict()
    weights = np.array([(r_map.get(m, 3.0) - 2.5) / 2.5 for m in rated_genres.index])
    
    g_matrix = rated_genres.values
    weighted_sum = np.dot(weights, g_matrix)
    g_counts = np.sum(g_matrix, axis=0)
    
    profile = np.zeros(len(genre_cols), dtype=float)
    for i in range(len(genre_cols)):
        if g_counts[i] > 0:
            profile[i] = weighted_sum[i] / g_counts[i]
        else:
            profile[i] = 0.0

    profile = np.clip((profile + 1.0) / 2.0, 0.0, 1.0)
    return profile


def prepare_interaction_vector(user_profile: np.ndarray, movie_vector: np.ndarray) -> np.ndarray:
    """
    Combines user profile and movie features into interaction vector (alignment, gap, profiles).
    """
    n_g = len(user_profile)
    m_genres = movie_vector[:n_g]
    
    alignment = user_profile * m_genres
    diff = np.abs(user_profile - m_genres)
    combined = np.concatenate([alignment, diff, user_profile, m_genres])
    return combined


def reduce_features_pca(
    X_raw: np.ndarray,
    pca_model: PCA = None,
    scaler: MinMaxScaler = None,
    angle_scaler: MinMaxScaler = None,
    n_components: int = PCA_N_COMPONENTS
) -> Tuple[np.ndarray, PCA, MinMaxScaler, MinMaxScaler]:
    """
    Reduces interaction feature dimension down to 4 features mapped to quantum angle range [0, pi].
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw)

    if pca_model is None:
        pca_model = PCA(n_components=n_components, random_state=42)
        X_pca = pca_model.fit_transform(X_scaled)
    else:
        X_pca = pca_model.transform(X_scaled)

    if angle_scaler is None:
        angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_quantum = angle_scaler.fit_transform(X_pca)
    else:
        X_quantum = angle_scaler.transform(X_pca)

    return X_quantum, pca_model, scaler, angle_scaler
