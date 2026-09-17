"""
Ablation Study Module (Research Grade).

Executes systematic ablation experiments evaluating:
1. Qubit Scaling N in {2, 4, 6}
2. Layer Depth L in {1, 2, 3}
3. Entanglement (CNOT Ring ON vs OFF)
4. Model Architectural Components
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from config import COBYLA_MAX_ITER
from src.preprocessing import (
    build_user_profile_historical,
    build_interaction_features,
    fit_preprocessing_pipeline,
    transform_preprocessing_pipeline
)
from src.quantum_model import VariationalQuantumCircuit, VQCOptimizer
from src.evaluation import evaluate_model_ranking


def run_ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genre_features_df: pd.DataFrame,
    all_candidate_mids: List[int],
    max_iter: int = 20
) -> pd.DataFrame:
    """
    Runs automated ablation experiments across qubit counts, layer depths, and entangling structures.
    """
    print("\n--- Running Ablation Experiments ---")
    
    unique_train_users = train_df["userId"].unique()
    train_user_profiles = {uid: build_user_profile_historical(uid, train_df, genre_features_df) for uid in unique_train_users}

    sample_train = train_df.sample(n=min(200, len(train_df)), random_state=42)
    X_train_raw, y_train_list = [], []
    for _, row in sample_train.iterrows():
        uid, mid, rating = int(row["userId"]), int(row["movieId"]), float(row["rating"])
        if mid in genre_features_df.index:
            u_prof = train_user_profiles[uid]
            m_prof = genre_features_df.loc[mid].values.astype(float)
            inter_feat = build_interaction_features(u_prof, m_prof)
            X_train_raw.append(inter_feat)
            y_train_list.append(1.0 if rating >= 4.0 else 0.0)

    X_train_raw = np.array(X_train_raw)
    y_train = np.array(y_train_list)

    ablation_configs = [
        {"name": "2-Qubits, 1-Layer, Entangled", "n_qubits": 2, "n_layers": 1, "entanglement": True},
        {"name": "2-Qubits, 2-Layers, Entangled", "n_qubits": 2, "n_layers": 2, "entanglement": True},
        {"name": "4-Qubits, 1-Layer, Entangled", "n_qubits": 4, "n_layers": 1, "entanglement": True},
        {"name": "4-Qubits, 2-Layers, Entangled (Default)", "n_qubits": 4, "n_layers": 2, "entanglement": True},
        {"name": "4-Qubits, 3-Layers, Entangled", "n_qubits": 4, "n_layers": 3, "entanglement": True},
        {"name": "4-Qubits, 2-Layers, No Entanglement", "n_qubits": 4, "n_layers": 2, "entanglement": False},
        {"name": "6-Qubits, 2-Layers, Entangled", "n_qubits": 6, "n_layers": 2, "entanglement": True},
    ]

    results_list = []

    for cfg in ablation_configs:
        print(f"  [Ablation] Testing Configuration: {cfg['name']}...")
        n_q = cfg["n_qubits"]
        n_l = cfg["n_layers"]
        ent = cfg["entanglement"]

        X_tr_q, pca, scaler, angle_scaler = fit_preprocessing_pipeline(X_train_raw, n_components=n_q)
        vqc = VariationalQuantumCircuit(n_qubits=n_q, n_layers=n_l, entanglement=ent)
        opt = VQCOptimizer(vqc, max_iter=max_iter)
        opt_theta, _ = opt.fit(X_tr_q, y_train, verbose=False)

        def score_func(uids, mids):
            preds = []
            for u, m in zip(uids, mids):
                u_p = train_user_profiles.get(u, np.full(19, 0.5))
                m_p = genre_features_df.loc[m].values.astype(float) if m in genre_features_df.index else np.zeros(19)
                raw_feat = build_interaction_features(u_p, m_p).reshape(1, -1)
                q_feat = transform_preprocessing_pipeline(raw_feat, scaler, pca, angle_scaler)[0]
                preds.append(vqc.evaluate_preference(q_feat, opt_theta))
            return np.array(preds)

        metrics = evaluate_model_ranking(score_func, test_df, all_candidate_mids, k_list=[5, 10], max_eval_users=50, max_candidates=100)

        results_list.append({
            "configuration": cfg["name"],
            "n_qubits": n_q,
            "n_layers": n_l,
            "entanglement": "Yes" if ent else "No",
            "num_params": vqc.num_params,
            "precision@5": metrics.get("precision@5", 0.0),
            "recall@5": metrics.get("recall@5", 0.0),
            "ndcg@5": metrics.get("ndcg@5", 0.0),
            "hit_rate@5": metrics.get("hit_rate@5", 0.0),
            "precision@10": metrics.get("precision@10", 0.0),
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "rmse": metrics.get("rmse", 0.0)
        })

    df_ablation = pd.DataFrame(results_list)
    return df_ablation
