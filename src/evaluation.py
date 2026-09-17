"""
Recommender Evaluation Framework Module (Research Grade).

Calculates standard top-K recommendation ranking metrics:
Precision@K, Recall@K, NDCG@K, Hit Rate@K, alongside RMSE and MAE.
Derived strictly from actual test set interactions without manual/invented entries.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Tuple

from config import PREFERENCE_THRESHOLD, EVAL_TOP_K_LIST


def calculate_precision_at_k(recommended_ids: List[int], relevant_set: Set[int], k: int) -> float:
    """Precision@K = |Recommended@K intersect Relevant| / K"""
    if k == 0:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = sum(1 for m in rec_k if m in relevant_set)
    return float(hits / k)


def calculate_recall_at_k(recommended_ids: List[int], relevant_set: Set[int], k: int) -> float:
    """Recall@K = |Recommended@K intersect Relevant| / |Relevant|"""
    if len(relevant_set) == 0:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = sum(1 for m in rec_k if m in relevant_set)
    return float(hits / len(relevant_set))


def calculate_ndcg_at_k(recommended_ids: List[int], relevant_set: Set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K).
    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    IDCG@K = sum_{i=1}^min(K, |Relevant|) 1 / log2(i + 1)
    """
    if k == 0 or len(relevant_set) == 0:
        return 0.0

    rec_k = recommended_ids[:k]
    dcg = 0.0
    for i, m in enumerate(rec_k):
        if m in relevant_set:
            dcg += 1.0 / math.log2(i + 2)

    idcg = 0.0
    n_rel = min(k, len(relevant_set))
    for i in range(n_rel):
        idcg += 1.0 / math.log2(i + 2)

    return float(dcg / idcg) if idcg > 0 else 0.0


def calculate_hit_rate_at_k(recommended_ids: List[int], relevant_set: Set[int], k: int) -> float:
    """Hit Rate@K = 1 if |Recommended@K intersect Relevant| > 0 else 0"""
    if len(relevant_set) == 0:
        return 0.0
    rec_k = recommended_ids[:k]
    hit = any(m in relevant_set for m in rec_k)
    return 1.0 if hit else 0.0


def calculate_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Calculates Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return round(rmse, 4), round(mae, 4)


def evaluate_model_ranking(
    score_predict_func,
    test_ratings_df: pd.DataFrame,
    all_candidate_mids: List[int],
    k_list: List[int] = EVAL_TOP_K_LIST,
    threshold: float = PREFERENCE_THRESHOLD,
    max_eval_users: int = 100,
    max_candidates: int = 200
) -> Dict[str, float]:
    """
    Evaluates a recommender model across test users for Precision@K, Recall@K, NDCG@K, Hit Rate@K, RMSE, MAE.
    Uses efficient candidate sampling (100 users, 200 candidates per user) for research scalability.
    """
    test_users = test_ratings_df["userId"].unique()
    
    relevant_by_user: Dict[int, Set[int]] = {}
    for uid, group in test_ratings_df.groupby("userId"):
        rel_mids = set(group[group["rating"] >= threshold]["movieId"].values)
        if len(rel_mids) > 0:
            relevant_by_user[uid] = rel_mids

    eval_users = [u for u in test_users if u in relevant_by_user]
    
    if len(eval_users) > max_eval_users:
        np.random.seed(42)
        eval_users = list(np.random.choice(eval_users, size=max_eval_users, replace=False))
    
    if len(eval_users) == 0:
        return {f"precision@{k}": 0.0 for k in k_list}

    metrics_accumulator: Dict[str, List[float]] = {}
    for k in k_list:
        metrics_accumulator[f"precision@{k}"] = []
        metrics_accumulator[f"recall@{k}"] = []
        metrics_accumulator[f"ndcg@{k}"] = []
        metrics_accumulator[f"hit_rate@{k}"] = []

    y_true_all, y_pred_all = [], []

    for uid in eval_users:
        rel_set = relevant_by_user[uid]
        
        # Subsample candidate set containing relevant items + random negative candidates
        if len(all_candidate_mids) > max_candidates:
            non_rel = [m for m in all_candidate_mids if m not in rel_set]
            n_neg = max(0, max_candidates - len(rel_set))
            np.random.seed(42 + uid)
            neg_sample = list(np.random.choice(non_rel, size=min(n_neg, len(non_rel)), replace=False)) if non_rel else []
            cand_pool = list(rel_set) + neg_sample
        else:
            cand_pool = all_candidate_mids

        # Predict score for sampled candidate movies for this user
        user_ids = [uid] * len(cand_pool)
        scores = score_predict_func(user_ids, cand_pool)
        
        # Rank movies by score descending
        ranked_indices = np.argsort(-scores)
        ranked_mids = [cand_pool[idx] for idx in ranked_indices]

        # Calculate metrics per K
        for k in k_list:
            p_k = calculate_precision_at_k(ranked_mids, rel_set, k)
            r_k = calculate_recall_at_k(ranked_mids, rel_set, k)
            n_k = calculate_ndcg_at_k(ranked_mids, rel_set, k)
            h_k = calculate_hit_rate_at_k(ranked_mids, rel_set, k)
            
            metrics_accumulator[f"precision@{k}"].append(p_k)
            metrics_accumulator[f"recall@{k}"].append(r_k)
            metrics_accumulator[f"ndcg@{k}"].append(n_k)
            metrics_accumulator[f"hit_rate@{k}"].append(h_k)

        # Collect rating predictions for test items
        user_test = test_ratings_df[test_ratings_df["userId"] == uid]
        test_mids = user_test["movieId"].values
        test_ratings = user_test["rating"].values
        
        test_uids = [uid] * len(test_mids)
        test_scores = score_predict_func(test_uids, list(test_mids))
        
        scaled_preds = 1.0 + 4.0 * test_scores
        y_true_all.extend(test_ratings)
        y_pred_all.extend(scaled_preds)

    results = {}
    for metric_name, val_list in metrics_accumulator.items():
        results[metric_name] = round(float(np.mean(val_list)), 4)

    if len(y_true_all) > 0:
        rmse, mae = calculate_rmse_mae(np.array(y_true_all), np.array(y_pred_all))
        results["rmse"] = rmse
        results["mae"] = mae
    else:
        results["rmse"] = 0.0
        results["mae"] = 0.0

    return results
