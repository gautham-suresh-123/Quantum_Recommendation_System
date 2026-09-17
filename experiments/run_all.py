"""
Master Automated Experiment Runner Module (Research Grade).

Executes the complete end-to-end research pipeline:
1. Loads official MovieLens 100K dataset & computes exact un-hardcoded dataset statistics.
2. Performs temporal split (Train / Val / Test).
3. Fits preprocessing pipeline (Scaler, PCA, AngleScaler) strictly on training set.
4. Trains Classical Baselines (Popularity, Content-Based, FunkSVD).
5. Trains VQC model & Hybrid VQC model.
6. Evaluates all models across Precision@K, Recall@K, NDCG@K, Hit Rate@K, RMSE, MAE.
7. Runs Hyperparameter / Architectural Ablation study.
8. Runs Quantum Shot & Noise sensitivity analysis.
9. Computes exact circuit complexity metrics.
10. Exports raw results to `results/*.csv` and `results/experiment_config.json`.
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

from config import (
    MODEL_FILE,
    RESULTS_DIR,
    N_QUBITS,
    VQC_LAYERS,
    PREFERENCE_THRESHOLD,
    RANDOM_SEED,
    EXPERIMENTAL_SEEDS,
    COBYLA_MAX_ITER,
    HYBRID_ALPHA,
    EVAL_TOP_K_LIST
)
from src.preprocessing import (
    download_and_load_data,
    get_dataset_statistics,
    encode_genres,
    temporal_train_test_split,
    build_user_profile_historical,
    build_interaction_features,
    fit_preprocessing_pipeline,
    transform_preprocessing_pipeline
)
from src.quantum_model import VariationalQuantumCircuit, VQCOptimizer, get_circuit_metrics
from src.baselines import PopularityBaseline, ContentBasedBaseline, MatrixFactorizationBaseline
from src.evaluation import evaluate_model_ranking
from experiments.ablation import run_ablation_study
from experiments.noise_experiment import run_shot_and_noise_experiments


def run_full_experiment_pipeline() -> Dict[str, Any]:
    start_time = time.time()
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("  QUANTUM RECOMMENDATION SYSTEM — MASTER EXPERIMENTAL PIPELINE")
    print("=" * 70)

    # Step 1: Load Data & Extract Dynamic Statistics
    print("\n[Step 1] Loading MovieLens 100K Dataset & Calculating Dynamic Statistics...")
    movies_df, ratings_df = download_and_load_data()
    genre_features_df = encode_genres(movies_df)
    dataset_stats = get_dataset_statistics(movies_df, ratings_df)
    
    print("  Dataset Statistics:")
    print(f"   - Name        : {dataset_stats['dataset_name']}")
    print(f"   - Users       : {dataset_stats['num_users']}")
    print(f"   - Movies      : {dataset_stats['num_movies']}")
    print(f"   - Ratings     : {dataset_stats['num_ratings']}")
    print(f"   - Rating Scale: {dataset_stats['rating_scale']} (Mean: {dataset_stats['mean_rating']})")

    # Step 2: Temporal Split
    print("\n[Step 2] Performing Leakage-Free Temporal Split (70% Train / 15% Val / 15% Test)...")
    train_df, val_df, test_df = temporal_train_test_split(ratings_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print(f"   - Train Split: {len(train_df)} interactions")
    print(f"   - Val Split  : {len(val_df)} interactions")
    print(f"   - Test Split : {len(test_df)} interactions")

    # Step 3: Fit Preprocessing Pipeline strictly on Training Set
    print("\n[Step 3] Building Historical User Profiles & Fitting Train-Only PCA Scaler Pipeline...")
    unique_train_users = train_df["userId"].unique()
    train_user_profiles = {uid: build_user_profile_historical(uid, train_df, genre_features_df) for uid in unique_train_users}

    # Sample interactions for VQC optimization
    sample_train = train_df.sample(n=min(400, len(train_df)), random_state=RANDOM_SEED)
    X_train_raw, y_train_list = [], []
    for _, row in sample_train.iterrows():
        uid, mid, rating = int(row["userId"]), int(row["movieId"]), float(row["rating"])
        if mid in genre_features_df.index:
            u_prof = train_user_profiles[uid]
            m_prof = genre_features_df.loc[mid].values.astype(float)
            inter_feat = build_interaction_features(u_prof, m_prof)
            X_train_raw.append(inter_feat)
            y_train_list.append(1.0 if rating >= PREFERENCE_THRESHOLD else 0.0)

    X_train_raw = np.array(X_train_raw)
    y_train = np.array(y_train_list)

    X_train_quantum, pca_model, scaler, angle_scaler = fit_preprocessing_pipeline(X_train_raw, n_components=N_QUBITS)
    print(f"   - Canonical Input Dim: {X_train_raw.shape[1]}D")
    print(f"   - Reduced Quantum Dim: {X_train_quantum.shape[1]} Qubits mapped to [0, pi]")

    # Step 4: Fit Classical Baselines
    print("\n[Step 4] Fitting Classical Baseline Models...")
    pop_model = PopularityBaseline()
    pop_model.fit(train_df)

    content_model = ContentBasedBaseline(genre_features_df)
    content_model.fit(train_df)

    mf_model = MatrixFactorizationBaseline(n_components=10)
    mf_model.fit(train_df)

    # Step 5: Train VQC Model
    print(f"\n[Step 5] Training {N_QUBITS}-Qubit VQC Model (Layers={VQC_LAYERS}, MaxIter={COBYLA_MAX_ITER})...")
    vqc = VariationalQuantumCircuit(n_qubits=N_QUBITS, n_layers=VQC_LAYERS, entanglement=True)
    optimizer = VQCOptimizer(vqc, max_iter=COBYLA_MAX_ITER, seed=RANDOM_SEED)
    optimal_theta, loss_history = optimizer.fit(X_train_quantum, y_train, verbose=False)
    print(f"   - Optimization Complete. Initial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")

    # Step 6: Evaluate All Models on Test Set
    print("\n[Step 6] Evaluating Models on Test Set across Precision@K, Recall@K, NDCG@K, Hit Rate@K...")
    all_candidate_mids = movies_df["movieId"].values.tolist()

    models_to_evaluate = {
        "Popularity Baseline": lambda uids, mids: pop_model.predict_batch(uids, mids),
        "Content-Based Baseline": lambda uids, mids: content_model.predict_batch(uids, mids),
        "Matrix Factorization (SVD)": lambda uids, mids: mf_model.predict_batch(uids, mids),
    }

    # Quantum VQC standalone prediction function
    def vqc_score_func(uids, mids):
        preds = []
        for u, m in zip(uids, mids):
            u_p = train_user_profiles.get(u, np.full(19, 0.5))
            m_p = genre_features_df.loc[m].values.astype(float) if m in genre_features_df.index else np.zeros(19)
            raw_feat = build_interaction_features(u_p, m_p).reshape(1, -1)
            q_feat = transform_preprocessing_pipeline(raw_feat, scaler, pca_model, angle_scaler)[0]
            preds.append(vqc.evaluate_preference(q_feat, optimal_theta))
        return np.array(preds)

    # Hybrid VQC + Content prediction function
    def hybrid_score_func(uids, mids):
        s_q = vqc_score_func(uids, mids)
        s_c = content_model.predict_batch(uids, mids)
        return HYBRID_ALPHA * s_q + (1.0 - HYBRID_ALPHA) * s_c

    models_to_evaluate["VQC Model (Quantum Only)"] = vqc_score_func
    models_to_evaluate["Hybrid VQC Model (Proposed)"] = hybrid_score_func

    metrics_results = []
    for model_name, score_fn in models_to_evaluate.items():
        print(f"   - Evaluating {model_name}...")
        res = evaluate_model_ranking(score_fn, test_df, all_candidate_mids, k_list=EVAL_TOP_K_LIST)
        res["model"] = model_name
        metrics_results.append(res)

    df_metrics = pd.DataFrame(metrics_results)
    cols = ["model", "precision@5", "recall@5", "ndcg@5", "hit_rate@5", "precision@10", "recall@10", "ndcg@10", "hit_rate@10", "rmse", "mae"]
    df_metrics = df_metrics[[c for c in cols if c in df_metrics.columns]]

    # Step 7: Circuit Complexity Analysis
    print("\n[Step 7] Computing Exact Quantum Circuit Complexity Metrics...")
    circuit_metrics = get_circuit_metrics(vqc)
    df_circuit = pd.DataFrame([circuit_metrics])
    print(f"   - Qubits: {circuit_metrics['n_qubits']}, Depth: {circuit_metrics['circuit_depth']}, Total Gates: {circuit_metrics['total_gates']}, CNOTs: {circuit_metrics['cnot_gates']}")

    # Step 8: Run Ablation Study
    print("\n[Step 8] Running Hyperparameter & Architectural Ablation Framework...")
    df_ablation = run_ablation_study(train_df, test_df, genre_features_df, all_candidate_mids, max_iter=25)

    # Step 9: Run Shot & Noise Analysis
    print("\n[Step 9] Running Quantum Shot & Noise Sensitivity Analysis...")
    df_noise = run_shot_and_noise_experiments(vqc, optimal_theta, scaler, pca_model, angle_scaler, test_df, genre_features_df, all_candidate_mids)

    # Step 10: Save Results to CSV & Model Artifact
    print("\n[Step 10] Saving All Empirical Results to results/*.csv and Model Artifact...")
    RESULTS_DIR.mkdir(exist_ok=True)
    df_metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    df_circuit.to_csv(RESULTS_DIR / "circuit_metrics.csv", index=False)
    df_ablation.to_csv(RESULTS_DIR / "ablation.csv", index=False)
    df_noise.to_csv(RESULTS_DIR / "noise_experiment.csv", index=False)

    exp_config = {
        "dataset": dataset_stats,
        "random_seed": RANDOM_SEED,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "n_qubits": N_QUBITS,
        "vqc_layers": VQC_LAYERS,
        "hybrid_alpha": HYBRID_ALPHA,
        "cobyla_max_iter": COBYLA_MAX_ITER,
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # Save Model Artifact
    artifact = {
        "n_qubits": N_QUBITS,
        "n_layers": VQC_LAYERS,
        "optimal_theta": optimal_theta,
        "pca_model": pca_model,
        "scaler": scaler,
        "angle_scaler": angle_scaler,
        "loss_history": loss_history,
        "hybrid_alpha": HYBRID_ALPHA,
        "genre_list": genre_features_df.columns.tolist()
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(artifact, f)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"SUCCESS: Pipeline completed in {elapsed:.2f} seconds.")
    print(f"Results written to: {RESULTS_DIR}")
    print(f"Model saved to    : {MODEL_FILE}")
    print("=" * 70)

    return {
        "metrics": df_metrics,
        "circuit": df_circuit,
        "ablation": df_ablation,
        "noise": df_noise
    }


if __name__ == "__main__":
    run_full_experiment_pipeline()
