"""
Simplified Model Training Script.

Trains the 4-Qubit Variational Quantum Circuit (VQC) offline.
Supports DEMO_MODE = True for fast viva demonstration (~2-3 seconds).
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from config import (
    MODEL_FILE,
    N_QUBITS,
    VQC_LAYERS,
    PREFERENCE_THRESHOLD,
    RANDOM_SEED,
    DEMO_MODE,
    DEMO_TRAIN_SAMPLES,
    FULL_TRAIN_SAMPLES,
    COBYLA_MAX_ITER
)
from preprocessing import (
    download_and_load_data,
    encode_genres,
    build_user_profile,
    prepare_interaction_vector,
    reduce_features_pca
)
from quantum_model import VariationalQuantumCircuit, VQCOptimizer


def train_model(
    demo_mode: bool = DEMO_MODE,
    save_path: str = str(MODEL_FILE)
) -> Dict[str, Any]:
    """
    Executes training pipeline for 4-qubit VQC recommendation model.
    """
    np.random.seed(RANDOM_SEED)
    max_samples = DEMO_TRAIN_SAMPLES if demo_mode else FULL_TRAIN_SAMPLES
    mode_name = "DEMO MODE (Fast Viva Run)" if demo_mode else "FULL MODE"
    
    print("=" * 60)
    print(f"  QUANTUM RECOMMENDER MODEL TRAINING - {mode_name}")
    print("=" * 60)

    # 1. Load Data
    print("\n[Step 1] Loading MovieLens dataset...")
    movies_df, ratings_df = download_and_load_data()
    genre_features_df = encode_genres(movies_df)

    # 2. Build User Profiles
    print("\n[Step 2] Building user preference profiles...")
    unique_users = ratings_df["userId"].unique()
    user_profiles = {uid: build_user_profile(uid, ratings_df, genre_features_df) for uid in unique_users}

    # 3. Sample Interaction Pairs
    print(f"\n[Step 3] Sampling {max_samples} interaction feature pairs...")
    valid_ratings = ratings_df[ratings_df["movieId"].isin(genre_features_df.index)].copy()
    if len(valid_ratings) > max_samples:
        valid_ratings = valid_ratings.sample(n=max_samples, random_state=RANDOM_SEED)

    X_list, y_list = [], []
    for idx, row in valid_ratings.iterrows():
        uid, mid, rating = int(row["userId"]), int(row["movieId"]), float(row["rating"])
        user_prof = user_profiles[uid]
        movie_vec = genre_features_df.loc[mid].values.astype(float)
        
        inter_vec = prepare_interaction_vector(user_prof, movie_vec)
        X_list.append(inter_vec)
        y_list.append(1.0 if rating >= PREFERENCE_THRESHOLD else 0.0)

    X = np.array(X_list)
    y = np.array(y_list)

    # 4. Train / Val Split & PCA Reduction
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\n[Step 4] Fitting PCA (38D -> {N_QUBITS} Qubits mapped to [0, pi])...")
    X_train_quantum, pca_model, scaler, angle_scaler = reduce_features_pca(X_train_raw, n_components=N_QUBITS)
    X_val_quantum, _, _, _ = reduce_features_pca(
        X_val_raw, pca_model=pca_model, scaler=scaler, angle_scaler=angle_scaler, n_components=N_QUBITS
    )

    # 5. Initialize VQC and Train
    print(f"\n[Step 5] Initializing {N_QUBITS}-Qubit VQC (Layers={VQC_LAYERS})...")
    vqc = VariationalQuantumCircuit(n_qubits=N_QUBITS, n_layers=VQC_LAYERS)
    optimizer = VQCOptimizer(vqc, max_iter=COBYLA_MAX_ITER)

    opt_theta, loss_history = optimizer.fit(X_train_quantum, y_train, verbose=True)

    # 6. Evaluation
    val_preds_prob = vqc.evaluate_batch(X_val_quantum, opt_theta)
    val_preds_bin = (val_preds_prob >= 0.5).astype(float)

    val_acc = float(accuracy_score(y_val, val_preds_bin))
    val_prec = float(precision_score(y_val, val_preds_bin, zero_division=0))
    val_rec = float(recall_score(y_val, val_preds_bin, zero_division=0))

    print(f"\n[Step 6] Validation Metrics:")
    print(f"  - Accuracy  : {val_acc * 100:.2f}%")
    print(f"  - Precision : {val_prec:.4f}")
    print(f"  - Recall    : {val_rec:.4f}")

    # 7. Save Artifact
    artifact = {
        "n_qubits": N_QUBITS,
        "n_layers": VQC_LAYERS,
        "optimal_theta": opt_theta,
        "pca_model": pca_model,
        "scaler": scaler,
        "angle_scaler": angle_scaler,
        "loss_history": loss_history,
        "val_acc": val_acc,
        "val_prec": val_prec,
        "val_rec": val_rec
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n[Step 7] Model saved successfully to {save_path}")
    print("=" * 60)
    return artifact


if __name__ == "__main__":
    train_model(demo_mode=DEMO_MODE)
