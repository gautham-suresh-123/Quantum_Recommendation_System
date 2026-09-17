"""
Data Preprocessing & Feature Pipeline Module (Research Grade).

Handles MovieLens 100K data loading, dynamic dataset auditing, multi-hot genre encoding,
temporal data splitting, leakage-free historical user profile construction, 76D canonical interaction feature extraction,
and strict train-only PCA/Scaler fitting.
"""

import os
import io
import zipfile
import logging
import requests
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_DIR,
    MOVIES_FILE,
    RATINGS_FILE,
    USERS_FILE,
    MOVIELENS_100K_URL,
    ALL_GENRES,
    PCA_N_COMPONENTS,
    CANONICAL_FEATURE_DIM
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
                item_cols = ["movieId", "title", "release_date", "video_release_date", "IMDb_URL"] + ALL_GENRES
                with z.open("ml-100k/u.item") as f:
                    item_df = pd.read_csv(f, sep="|", encoding="latin-1", names=item_cols)
                
                def get_genre_str(row):
                    active = [g for g in ALL_GENRES if row[g] == 1]
                    return "|".join(active) if active else "Unknown"
                
                item_df["genres"] = item_df.apply(get_genre_str, axis=1)
                item_df[["movieId", "title", "genres"]].to_csv(MOVIES_FILE, index=False)

                with z.open("ml-100k/u.data") as f:
                    ratings_df = pd.read_csv(f, sep="\t", names=["userId", "movieId", "rating", "timestamp"])
                ratings_df.to_csv(RATINGS_FILE, index=False)
                logger.info("MovieLens dataset downloaded and saved successfully.")
        except Exception as e:
            logger.error(f"Failed to download MovieLens dataset: {e}")
            raise FileNotFoundError("Dataset not found. Please download the required dataset before running the experiment.")

    movies_df = pd.read_csv(MOVIES_FILE)
    ratings_df = pd.read_csv(RATINGS_FILE)
    
    movies_df.drop_duplicates(subset=["movieId"], inplace=True)
    movies_df["genres"] = movies_df["genres"].fillna("Unknown")
    
    ratings_df.dropna(subset=["userId", "movieId", "rating", "timestamp"], inplace=True)
    ratings_df.drop_duplicates(subset=["userId", "movieId"], inplace=True)
    
    return movies_df, ratings_df


def get_dataset_statistics(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes exact, un-hardcoded dataset statistics from the loaded data.
    """
    n_users = ratings_df["userId"].nunique()
    n_movies = movies_df["movieId"].nunique()
    n_ratings = len(ratings_df)
    min_rating = float(ratings_df["rating"].min())
    max_rating = float(ratings_df["rating"].max())
    mean_rating = float(ratings_df["rating"].mean())
    
    timestamp_min = int(ratings_df["timestamp"].min())
    timestamp_max = int(ratings_df["timestamp"].max())
    
    genre_counts = {}
    for g in ALL_GENRES:
        count = movies_df["genres"].apply(lambda s: g in str(s).split("|")).sum()
        genre_counts[g] = int(count)

    return {
        "dataset_name": "MovieLens 100K",
        "num_users": n_users,
        "num_movies": n_movies,
        "num_ratings": n_ratings,
        "rating_scale": f"{min_rating:.1f} - {max_rating:.1f}",
        "mean_rating": round(mean_rating, 3),
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "genre_counts": genre_counts
    }


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


def temporal_train_test_split(
    ratings_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs leakage-free temporal splitting based on interaction timestamps.
    Earlier interactions -> Training
    Intermediate interactions -> Validation
    Latest interactions -> Testing
    """
    df_sorted = ratings_df.sort_values(by="timestamp").reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def build_user_profile_historical(
    user_id: int,
    ratings_history_df: pd.DataFrame,
    genre_features_df: pd.DataFrame,
    before_timestamp: Optional[int] = None,
    exclude_movie_id: Optional[int] = None
) -> np.ndarray:
    """
    Calculates leakage-free user genre preference vector P_u in [0, 1]^19 using ONLY historical ratings.
    If before_timestamp is provided, filters ratings strictly occurring before that timestamp.
    If exclude_movie_id is provided, excludes the target movie interaction to prevent target leakage.
    """
    u_ratings = ratings_history_df[ratings_history_df["userId"] == user_id]
    
    if before_timestamp is not None:
        u_ratings = u_ratings[u_ratings["timestamp"] < before_timestamp]
        
    if exclude_movie_id is not None:
        u_ratings = u_ratings[u_ratings["movieId"] != exclude_movie_id]
        
    genre_cols = [g for g in ALL_GENRES if g in genre_features_df.columns]
    
    if len(u_ratings) == 0:
        return np.full(len(genre_cols), 0.5, dtype=float)

    user_mids = u_ratings["movieId"].values
    rated_genres = genre_features_df.loc[genre_features_df.index.isin(user_mids), genre_cols]
    
    if len(rated_genres) == 0:
        return np.full(len(genre_cols), 0.5, dtype=float)

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


def build_interaction_features(user_profile: np.ndarray, movie_profile: np.ndarray) -> np.ndarray:
    """
    CANONICAL FEATURE GENERATION FUNCTION (76D):
    Used IDENTICALLY across training, validation, testing, and inference.
    X_ui = [User Profile * Movie Profile (19D), |User Profile - Movie Profile| (19D), User Profile (19D), Movie Profile (19D)]
    """
    user_profile = np.asarray(user_profile, dtype=float)
    movie_profile = np.asarray(movie_profile, dtype=float)
    
    alignment = user_profile * movie_profile
    abs_diff = np.abs(user_profile - movie_profile)
    combined = np.concatenate([alignment, abs_diff, user_profile, movie_profile])
    return combined


def fit_preprocessing_pipeline(
    X_train_raw: np.ndarray,
    n_components: int = PCA_N_COMPONENTS
) -> Tuple[np.ndarray, PCA, MinMaxScaler, MinMaxScaler]:
    """
    Fits Scaler, PCA, and Angle Scaler ONLY on training data.
    """
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_scaled = scaler.fit_transform(X_train_raw)

    pca_model = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca_model.fit_transform(X_train_scaled)

    angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_quantum = angle_scaler.fit_transform(X_train_pca)

    return X_train_quantum, pca_model, scaler, angle_scaler


def transform_preprocessing_pipeline(
    X_raw: np.ndarray,
    scaler: MinMaxScaler,
    pca_model: PCA,
    angle_scaler: MinMaxScaler
) -> np.ndarray:
    """
    Transforms validation / test / inference feature matrices strictly using PRE-FITTED transformers.
    Never fits on test/inference data.
    """
    X_scaled = scaler.transform(X_raw)
    X_pca = pca_model.transform(X_scaled)
    X_quantum = angle_scaler.transform(X_pca)
    return X_quantum
