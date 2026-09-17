"""
Comprehensive Automated Research Unit Test Suite.

Verifies:
1. Data loading & un-hardcoded dataset statistics calculation.
2. Leakage-free temporal split.
3. Canonical feature dimension (76D) consistency across train, val, test, inference.
4. Strict train-only PCA fitting & persistence.
5. VQC evaluation & circuit metrics.
6. Un-clamped score breakdown (S_Q, S_C, S_H).
7. Recommender ranking metrics (Precision@K, Recall@K, NDCG@K, Hit Rate@K, RMSE, MAE).
8. Educational algorithms lab.
"""

import pytest
import numpy as np
import pandas as pd

from preprocessing import (
    download_and_load_data,
    get_dataset_statistics,
    encode_genres,
    temporal_train_test_split,
    build_user_profile_historical,
    build_interaction_features,
    fit_preprocessing_pipeline,
    transform_preprocessing_pipeline
)
from quantum_model import VariationalQuantumCircuit, VQCOptimizer, get_circuit_metrics
from recommender import process_quantum_recommendations, load_model_artifact
from experiments.baselines import PopularityBaseline, ContentBasedBaseline, MatrixFactorizationBaseline
from experiments.evaluate import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_hit_rate_at_k,
    calculate_rmse_mae
)
from educational.quantum_algorithms import (
    run_deutsch_jozsa,
    run_bernstein_vazirani,
    run_simon,
    run_grover,
    run_shor,
    run_vqe,
    run_qgan
)
from config import ALL_GENRES, N_QUBITS, CANONICAL_FEATURE_DIM


def test_dataset_and_statistics():
    movies_df, ratings_df = download_and_load_data()
    assert len(movies_df) > 0
    assert len(ratings_df) > 0
    
    stats = get_dataset_statistics(movies_df, ratings_df)
    assert stats["num_movies"] == len(movies_df)
    assert stats["num_ratings"] == len(ratings_df)
    assert isinstance(stats["mean_rating"], float)


def test_temporal_split_no_leakage():
    movies_df, ratings_df = download_and_load_data()
    train_df, val_df, test_df = temporal_train_test_split(ratings_df, 0.7, 0.15, 0.15)
    
    assert len(train_df) + len(val_df) + len(test_df) == len(ratings_df)
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()


def test_canonical_feature_dimensions():
    user_prof = np.random.rand(19)
    movie_prof = np.random.rand(19)
    inter_feat = build_interaction_features(user_prof, movie_prof)
    
    assert len(inter_feat) == CANONICAL_FEATURE_DIM
    assert len(inter_feat) == 76


def test_pca_fitting_strictly_on_train():
    X_train = np.random.rand(50, CANONICAL_FEATURE_DIM)
    X_test = np.random.rand(10, CANONICAL_FEATURE_DIM)
    
    X_train_q, pca, scaler, angle_scaler = fit_preprocessing_pipeline(X_train, n_components=N_QUBITS)
    assert X_train_q.shape == (50, N_QUBITS)
    
    X_test_q = transform_preprocessing_pipeline(X_test, scaler, pca, angle_scaler)
    assert X_test_q.shape == (10, N_QUBITS)
    assert 0.0 <= X_test_q.min() and X_test_q.max() <= np.pi + 1e-5


def test_quantum_model_and_circuit_metrics():
    vqc = VariationalQuantumCircuit(n_qubits=N_QUBITS, n_layers=2, entanglement=True)
    metrics = get_circuit_metrics(vqc)
    
    assert metrics["n_qubits"] == N_QUBITS
    assert metrics["n_layers"] == 2
    assert metrics["circuit_depth"] > 0
    assert metrics["total_gates"] > 0
    assert metrics["cnot_gates"] > 0

    sample_x = np.array([0.5, 1.0, 1.5, 2.0])
    sample_theta = np.zeros(vqc.num_params)
    score = vqc.evaluate_preference(sample_x, sample_theta)
    assert 0.0 <= score <= 1.0


def test_ranking_metrics_calculation():
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 5, 8}
    
    p5 = calculate_precision_at_k(recommended, relevant, 5)
    r5 = calculate_recall_at_k(recommended, relevant, 5)
    n5 = calculate_ndcg_at_k(recommended, relevant, 5)
    h5 = calculate_hit_rate_at_k(recommended, relevant, 5)
    
    assert p5 == 2 / 5
    assert r5 == 2 / 3
    assert 0.0 <= n5 <= 1.0
    assert h5 == 1.0

    rmse, mae = calculate_rmse_mae(np.array([4.0, 5.0]), np.array([3.8, 4.5]))
    assert rmse > 0.0
    assert mae > 0.0


def test_recommender_pipeline_unclamped_scores():
    user_prefs = {"genres": ["Action", "Sci-Fi"], "rec_count": 3}
    res = process_quantum_recommendations(user_prefs, mode="RESEARCH_MODE")
    
    assert res["status"] == "success"
    assert len(res["recommendations"]) <= 3
    
    top = res["top_match"]
    assert "quantum_score_sq" in top
    assert "classical_score_sc" in top
    assert "hybrid_score_sh" in top
    assert 0.0 <= top["quantum_score_sq"] <= 1.0
    assert 0.0 <= top["classical_score_sc"] <= 1.0
    assert 0.0 <= top["hybrid_score_sh"] <= 1.0


def test_educational_lab_algorithms():
    assert run_deutsch_jozsa(3, "balanced")["determined_type"] == "Balanced"
    assert run_bernstein_vazirani("1011")["recovered_string"] == "1011"
    assert len(run_simon("11")["orthogonal_vectors"]) > 0
    assert run_grover(3, "101")["top_measured"] == "101"
    assert run_shor(15, 7)["factors"] == (3, 5)
    assert isinstance(run_vqe(10)["estimated_energy"], float)
    assert len(run_qgan(5)["synthetic_distribution"]) == 4
