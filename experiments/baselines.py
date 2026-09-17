"""
Classical Recommendation Baselines Module (Research Grade).

Implements competitive classical benchmark models:
1. Popularity Baseline
2. Content-Based Cosine Similarity Baseline
3. Matrix Factorization (FunkSVD / Truncated SVD) Baseline
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from config import ALL_GENRES
from preprocessing import build_user_profile_historical


class PopularityBaseline:
    """
    Popularity-based recommendation baseline.
    Ranks items based on interaction frequency and mean rating in the training set.
    """

    def __init__(self):
        self.movie_scores: Dict[int, float] = {}

    def fit(self, train_ratings_df: pd.DataFrame):
        grouped = train_ratings_df.groupby("movieId").agg(
            count=("rating", "count"),
            mean_rating=("rating", "mean")
        )
        max_count = grouped["count"].max() if len(grouped) > 0 else 1.0
        
        for mid, row in grouped.iterrows():
            # Normalized popularity score combining frequency and mean rating
            freq_score = row["count"] / max_count
            rating_score = (row["mean_rating"] - 1.0) / 4.0
            self.movie_scores[mid] = float(0.5 * freq_score + 0.5 * rating_score)

    def predict_score(self, user_id: int, movie_id: int) -> float:
        return self.movie_scores.get(movie_id, 0.0)

    def predict_batch(self, user_ids: List[int], movie_ids: List[int]) -> np.ndarray:
        return np.array([self.predict_score(u, m) for u, m in zip(user_ids, movie_ids)])


class ContentBasedBaseline:
    """
    Content-Based Cosine Similarity baseline.
    Ranks items by cosine similarity between historical user genre profile and target movie genre vector.
    """

    def __init__(self, genre_features_df: pd.DataFrame):
        self.genre_features_df = genre_features_df
        self.user_profiles: Dict[int, np.ndarray] = {}

    def fit(self, train_ratings_df: pd.DataFrame):
        unique_users = train_ratings_df["userId"].unique()
        for uid in unique_users:
            self.user_profiles[uid] = build_user_profile_historical(uid, train_ratings_df, self.genre_features_df)

    def predict_score(self, user_id: int, movie_id: int) -> float:
        u_prof = self.user_profiles.get(user_id, np.full(len(ALL_GENRES), 0.5))
        if movie_id in self.genre_features_df.index:
            m_vec = self.genre_features_df.loc[movie_id].values.astype(float)
        else:
            m_vec = np.zeros(len(ALL_GENRES), dtype=float)

        sim = cosine_similarity(u_prof.reshape(1, -1), m_vec.reshape(1, -1))[0, 0]
        return float(np.clip(sim, 0.0, 1.0))

    def predict_batch(self, user_ids: List[int], movie_ids: List[int]) -> np.ndarray:
        return np.array([self.predict_score(u, m) for u, m in zip(user_ids, movie_ids)])


class MatrixFactorizationBaseline:
    """
    Matrix Factorization (Truncated SVD / Collaborative Filtering) baseline.
    Learns latent factor representations for users and items strictly from training interactions.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_ids: List[int] = []
        self.movie_ids: List[int] = []
        self.user_idx_map: Dict[int, int] = {}
        self.movie_idx_map: Dict[int, int] = {}
        self.user_factors: np.ndarray = np.array([])
        self.item_factors: np.ndarray = np.array([])
        self.global_mean: float = 0.5

    def fit(self, train_ratings_df: pd.DataFrame):
        self.user_ids = sorted(train_ratings_df["userId"].unique().tolist())
        self.movie_ids = sorted(train_ratings_df["movieId"].unique().tolist())
        
        self.user_idx_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_idx_map = {mid: j for j, mid in enumerate(self.movie_ids)}
        
        # Build user-item interaction matrix
        R = np.zeros((len(self.user_ids), len(self.movie_ids)), dtype=float)
        for _, row in train_ratings_df.iterrows():
            u_i = self.user_idx_map[row["userId"]]
            m_j = self.movie_idx_map[row["movieId"]]
            # Map rating to [0, 1] scale
            R[u_i, m_j] = (row["rating"] - 1.0) / 4.0

        self.global_mean = float(np.mean(R[R > 0])) if np.sum(R > 0) > 0 else 0.5

        # Fit SVD
        n_comp = min(self.n_components, min(R.shape) - 1)
        if n_comp >= 1:
            self.user_factors = self.svd.fit_transform(R)
            self.item_factors = self.svd.components_.T
        else:
            self.user_factors = np.zeros((len(self.user_ids), 1))
            self.item_factors = np.zeros((len(self.movie_ids), 1))

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if user_id in self.user_idx_map and movie_id in self.movie_idx_map:
            u_i = self.user_idx_map[user_id]
            m_j = self.movie_idx_map[movie_id]
            pred = np.dot(self.user_factors[u_i], self.item_factors[m_j])
            return float(np.clip(pred, 0.0, 1.0))
        else:
            return self.global_mean

    def predict_batch(self, user_ids: List[int], movie_ids: List[int]) -> np.ndarray:
        return np.array([self.predict_score(u, m) for u, m in zip(user_ids, movie_ids)])
