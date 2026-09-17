"""
Quantum Shot & Noise Analysis Module (Research Grade).

Evaluates VQC preference prediction under:
1. Ideal Statevector simulation
2. Finite-shot execution (1024, 4096, 8192 shots)
3. Simulated quantum noise (Depolarizing Noise & Readout Errors via Qiskit Aer)
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from quantum_model import VariationalQuantumCircuit
from preprocessing import (
    build_user_profile_historical,
    build_interaction_features,
    fit_preprocessing_pipeline,
    transform_preprocessing_pipeline
)
from experiments.evaluate import evaluate_model_ranking


def build_depolarizing_noise_model(p1: float = 0.01, p2: float = 0.02, p_readout: float = 0.02):
    """Constructs realistic NISQ depolarizing & readout error noise model via Qiskit Aer."""
    try:
        from qiskit_aer.noise import NoiseModel, DepolarizingError, ReadoutError
        noise_model = NoiseModel()
        
        err1 = DepolarizingError(p1, 1)
        noise_model.add_all_qubit_quantum_error(err1, ["ry", "rz", "h"])
        
        err2 = DepolarizingError(p2, 2)
        noise_model.add_all_qubit_quantum_error(err2, ["cx"])
        
        r_err = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        noise_model.add_all_qubit_readout_error(r_err)
        
        return noise_model
    except ImportError:
        return None


def run_shot_and_noise_experiments(
    vqc: VariationalQuantumCircuit,
    optimal_theta: np.ndarray,
    scaler: Any,
    pca: Any,
    angle_scaler: Any,
    test_df: pd.DataFrame,
    genre_features_df: pd.DataFrame,
    all_candidate_mids: List[int]
) -> pd.DataFrame:
    """
    Runs shot variance and quantum noise impact analysis.
    Only reports metrics produced by actual execution.
    """
    print("\n--- Running Shot & Noise Analysis ---")
    
    unique_test_users = test_df["userId"].unique()
    test_user_profiles = {uid: build_user_profile_historical(uid, test_df, genre_features_df) for uid in unique_test_users}

    experiments_to_run = [
        {"execution_mode": "Ideal Statevector", "shots": None, "noise_model": None},
        {"execution_mode": "Finite Shots (1024)", "shots": 1024, "noise_model": None},
        {"execution_mode": "Finite Shots (4096)", "shots": 4096, "noise_model": None},
        {"execution_mode": "Finite Shots (8192)", "shots": 8192, "noise_model": None},
    ]

    noise_model = build_depolarizing_noise_model()
    if noise_model is not None:
        experiments_to_run.append({"execution_mode": "Noisy Simulation (Depolarizing p=0.01)", "shots": 1024, "noise_model": noise_model})
    else:
        print("  [INFO] Qiskit Aer NoiseModel unavailable. Skipping noisy simulation.")

    results_list = []

    for exp in experiments_to_run:
        mode_label = exp["execution_mode"]
        shots_val = exp["shots"]
        noise_m = exp["noise_model"]
        print(f"  [Execution] Evaluating mode: {mode_label}...")

        def score_func(uids, mids):
            preds = []
            for u, m in zip(uids, mids):
                u_p = test_user_profiles.get(u, np.full(19, 0.5))
                m_p = genre_features_df.loc[m].values.astype(float) if m in genre_features_df.index else np.zeros(19)
                raw_feat = build_interaction_features(u_p, m_p).reshape(1, -1)
                q_feat = transform_preprocessing_pipeline(raw_feat, scaler, pca, angle_scaler)[0]
                prob = vqc.evaluate_preference(q_feat, optimal_theta, shots=shots_val, noise_model=noise_m)
                preds.append(prob)
            return np.array(preds)

        metrics = evaluate_model_ranking(score_func, test_df, all_candidate_mids, k_list=[5, 10], max_eval_users=30, max_candidates=50)

        results_list.append({
            "execution_mode": mode_label,
            "shots": shots_val if shots_val else "Exact",
            "noise_simulated": "Yes" if noise_m else "No",
            "precision@5": metrics.get("precision@5", 0.0),
            "recall@5": metrics.get("recall@5", 0.0),
            "ndcg@5": metrics.get("ndcg@5", 0.0),
            "precision@10": metrics.get("precision@10", 0.0),
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "rmse": metrics.get("rmse", 0.0)
        })

    return pd.DataFrame(results_list)
