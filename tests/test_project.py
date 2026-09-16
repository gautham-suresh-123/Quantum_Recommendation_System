"""
Consolidated Unit Test Suite for Quantum Recommendation Project.
"""

import pytest
import numpy as np
import pandas as pd

from preprocessing import (
    download_and_load_data,
    encode_genres,
    build_user_profile,
    prepare_interaction_vector,
    reduce_features_pca
)
from quantum_model import AngleEncoding, VariationalQuantumCircuit, VQCOptimizer
from recommender import QuantumRecommender, score_candidates, rank_movies
from quantum_algorithms import (
    run_deutsch_jozsa,
    run_bernstein_vazirani,
    run_simon,
    run_grover,
    run_shor,
    run_vqe,
    run_qgan
)
from config import ALL_GENRES, N_QUBITS


def test_preprocessing():
    movies_df, ratings_df = download_and_load_data()
    assert len(movies_df) > 0
    assert len(ratings_df) > 0
    
    genre_features_df = encode_genres(movies_df)
    assert len(genre_features_df) == len(movies_df)
    assert "Action" in genre_features_df.columns
    
    u_prof = build_user_profile(user_id=1, ratings_df=ratings_df, genre_features_df=genre_features_df)
    assert len(u_prof) == len(ALL_GENRES)
    assert 0.0 <= u_prof.min() <= u_prof.max() <= 1.0


def test_quantum_model():
    vqc = VariationalQuantumCircuit(n_qubits=N_QUBITS, n_layers=2)
    assert vqc.n_qubits == N_QUBITS
    assert vqc.num_params == 24
    
    sample_x = np.array([0.5, 1.0, 1.5, 2.0])
    sample_theta = np.zeros(vqc.num_params)
    score = vqc.evaluate_preference(sample_x, sample_theta)
    assert 0.0 <= score <= 1.0


def test_recommender_pipeline():
    rec = QuantumRecommender()
    top_5 = rec.recommend_movies(user_id=1, top_k=5)
    assert len(top_5) == 5
    assert "Rank" in top_5.columns
    assert "quantum_score" in top_5.columns
    assert top_5.iloc[0]["Rank"] == 1


def test_lab_algorithms():
    assert run_deutsch_jozsa(3, "balanced")["determined_type"] == "Balanced"
    assert run_bernstein_vazirani("1011")["recovered_string"] == "1011"
    assert len(run_simon("11")["orthogonal_vectors"]) > 0
    assert run_grover(3, "101")["top_measured"] == "101"
    assert run_shor(15, 7)["factors"] == (3, 5)
    assert isinstance(run_vqe(10)["estimated_energy"], float)
    assert len(run_qgan(5)["synthetic_distribution"]) == 4
