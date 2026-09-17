# Research Assumptions and Experimental Protocol Specifications

This document explicitly records all design decisions, mathematical definitions, dataset thresholds, and experimental protocol assumptions enforced within the Quantum Recommendation System research framework.

---

## 1. Dataset Assumptions
- **Benchmark Dataset**: MovieLens 100K dataset published by GroupLens Research.
- **Data Integrity**: Rows with missing `userId`, `movieId`, `rating`, or `timestamp` are removed. Duplicate interaction records are deduplicated based on `(userId, movieId)`.
- **Genre Attributes**: 19 standard MovieLens genres encoded into binary multi-hot vectors $M_i \in \{0, 1\}^{19}$.

---

## 2. Interaction & Preference Definitions
- **Rating Scale**: Ordinal numerical ratings from 1.0 to 5.0.
- **Positive Interaction Threshold**: Ratings $r_{ui} \ge 4.0$ are classified as positive preference interactions ($y_{ui} = 1.0$). Ratings $r_{ui} < 4.0$ are classified as non-preferred ($y_{ui} = 0.0$).
- **Rating Normalization**: Historical user rating weights are mapped using $w(r_{ui}) = (r_{ui} - 2.5) / 2.5 \in [-0.6, 1.0]$.

---

## 3. Data Leakage Prevention Protocol
- **Temporal Splitting**: Interactions are sorted chronologically by timestamp. The first 70% form the Training Set, the next 15% form the Validation Set, and the final 15% form the Test Set.
- **Historical User Profile Construction**: For any target evaluation interaction $(u, i, t)$, user profile $P_u$ is computed strictly using interactions $(u, i', t')$ where $t' < t$. Future interaction data is strictly prohibited from leaking into user profiles.

---

## 4. Feature Extraction & Preprocessing Pipeline
- **Canonical Feature Vector**:
  $$X_{ui} = \big[ P_u \odot M_i \,\,\|\,\, |P_u - M_i| \,\,\|\,\, P_u \,\,\|\,\, M_i \big] \in \mathbb{R}^{76}$$
- **Train-Only Preprocessing Fitting**:
  1. MinMaxScaler maps raw features $X_{\text{train}} \rightarrow [0, 1]^{76}$.
  2. Principal Component Analysis (PCA) reduces $[0, 1]^{76} \rightarrow \mathbb{R}^4$.
  3. Angle MinMaxScaler maps reduced features to $[0, \pi]^4$.
  Transformers are fitted **strictly** on $X_{\text{train}}$. Validation, testing, and inference samples are transformed using pre-fitted transformers.

---

## 5. Quantum Circuit & Hybrid Scoring
- **Qubit Encoding**: 4 features mapped to 4 qubits using single-qubit $R_y(Z_j)$ rotations on $|0\rangle^{\otimes 4}$.
- **Ansatz Structure**: 2 layers of $R_y(\theta)$ and $R_z(\theta)$ rotations with CNOT ring entanglers.
- **Optimizer**: SciPy COBYLA with maximum 50 iterations and global random seed 42.
- **Quantum Score ($S_Q$)**: Statevector measurement probability $P(|1\rangle)_0 \in [0.0, 1.0]$.
- **Classical Score ($S_C$)**: Cosine similarity between user profile $P_u$ and movie genre vector $M_i$.
- **Hybrid Score ($S_H$)**:
  $$S_H = \alpha S_Q + (1 - \alpha) S_C \quad (\text{default } \alpha = 0.5)$$
- **Score Clamping**: Artificial score clamping (e.g. `min(max(score, 0.5), 0.999)`) is strictly eliminated. Un-clamped, calibrated scores are exposed directly.

---

## 6. Evaluation Protocol
- **Metrics**: Precision@K, Recall@K, NDCG@K, Hit Rate@K evaluated at $K \in \{5, 10\}$, alongside RMSE and MAE.
- **Candidate Pool**: Full MovieLens catalog (1,682 candidate movies) used for ranking evaluation during Research Mode.
