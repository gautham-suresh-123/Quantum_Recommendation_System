# Empirical Investigation of Hybrid Quantum-Classical Movie Recommendation using Variational Quantum Circuits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.2](https://img.shields.io/badge/Qiskit-2.2-purple.svg)](https://qiskit.org/)

This repository implements a research-grade, fully reproducible experimental framework evaluating a **Hybrid Quantum-Classical Movie Recommendation System** using Variational Quantum Circuits (VQCs) on the MovieLens 100K benchmark dataset.

---

## 1. Abstract
Personalized recommendation systems face severe scalability and cold-start challenges in high-dimensional feature spaces. Variational Quantum Circuits (VQCs) offer a promising Noisy Intermediate-Scale Quantum (NISQ) paradigm for learning non-linear user-item preference interactions within compact Hilbert spaces. In this work, we evaluate a 4-qubit parameterized quantum ansatz against classical baselines (Popularity, Content-Based Cosine Similarity, Matrix Factorization/SVD). Historical user preference vectors and item genre profiles are combined into a canonical 76-dimensional interaction space, reduced via Principal Component Analysis (PCA) to $N=4$ dimensions, and mapped onto qubits via $R_y$ angle feature encoding. Enforcing leakage-free temporal splitting, we report un-hardcoded metrics for Precision@K, Recall@K, NDCG@K, Hit Rate@K, RMSE, and MAE derived strictly from execution outputs.

---

## 2. System Architecture

```text
MovieLens 100K Raw Data (u.data, u.item)
            │
            ▼
Leakage-Free Temporal Splitter (70% Train / 15% Val / 15% Test)
            │
            ▼
Historical User Profile Builder P_u ∈ [0, 1]^19
            │
            ▼
Canonical Interaction Vector X_ui = [P_u ⊙ M_i  ||  |P_u - M_i|  ||  P_u  ||  M_i] ∈ ℝ^76
            │
            ▼
Train-Only Preprocessing Pipeline (MinMaxScaler ──► PCA (4D) ──► AngleScaler [0, π])
            │
            ▼
4-Qubit Ry Angle Encoding Feature Map: |ψ_in⟩ = ⨂ Ry(z_j)|0⟩
            │
            ▼
4-Qubit 2-Layer VQC (Ry/Rz Rotations + CNOT Ring Entanglers)
            │
            ▼
Quantum Preference Measurement Probability S_Q = P(|1⟩)_0
            │
            ▼
Hybrid Score Calculation S_H = α S_Q + (1 - α) S_C
            │
            ▼
Top-K Ranking & Recommender Metric Evaluation
```

---

## 3. Methodological Rigor & Data Leakage Prevention
- **Temporal Splitting**: Interactions are sorted by timestamp ($t_{\text{train}} \le t_{\text{val}} \le t_{\text{test}}$) to ensure future interactions are never visible during historical profile construction.
- **Canonical Feature Generator**: Single unified representation `build_interaction_features(user_profile, movie_profile)` (76D) used identically across training, validation, testing, and inference.
- **Train-Only Preprocessing**: MinMaxScaler and PCA models are fitted **strictly** on the training split and persisted in `models/vqc_recommender.pkl`. Validation/test/inference feature vectors are transformed using pre-fitted transformers.
- **Zero Score Clamping**: Artificial clipping functions (`min(max(score, 0.5), 0.999)`) have been eliminated. Raw $S_Q$, $S_C$, and $S_H$ metrics are exposed directly.

---

## 4. Experimental Results

> [!NOTE]
> All numerical values below are automatically read from raw empirical result artifacts (`results/metrics.csv` and `results/circuit_metrics.csv`). Zero metrics are hardcoded.

### Table I: Empirical Model Performance Comparison (MovieLens 100K)

| Model | Precision@5 | Recall@5 | NDCG@5 | Hit Rate@5 | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | RMSE | MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| **Popularity Baseline** | 0.5660 | 0.1049 | 0.5786 | 0.7900 | 0.5090 | 0.1880 | 0.5473 | 0.8500 | 1.3523 | 1.1157 |
| **Content-Based Baseline** | 0.3520 | 0.0651 | 0.3550 | 0.6600 | 0.3120 | 0.1277 | 0.3412 | 0.7900 | 1.6717 | 1.4295 |
| **Matrix Factorization (SVD)** | 0.2300 | 0.0731 | 0.3429 | 0.8100 | 0.1580 | 0.0762 | 0.2686 | 0.8200 | 1.3628 | 1.0851 |
| **VQC Model (Quantum Only)** | 0.3120 | 0.0343 | 0.3234 | 0.5500 | 0.2950 | 0.0681 | 0.3077 | 0.6300 | 1.4109 | 1.1875 |
| **Hybrid VQC Model (Proposed)** | 0.3300 | 0.0572 | 0.3555 | 0.6100 | 0.3220 | 0.1325 | 0.3573 | 0.7700 | 1.5262 | 1.3029 |

### Table II: VQC Circuit Complexity Metrics

| Qubits ($N$) | Layers ($L$) | Entanglement | Trainable Parameters | Circuit Depth | Total Gates | CNOT Gates | Single-Qubit Gates |
|---|---|---|---|---|---|---|---|
| 4 | 2 | CNOT Ring | 24 | 15 | 36 | 8 | 28 |

---

## 5. Repository Structure

```text
Quantum_Recommendation_System/
│
├── config.py                 # Configuration parameters & global seeds
├── preprocessing.py          # Data loading, temporal split, canonical 76D feature extraction
├── quantum_model.py          # N-qubit VQC ansatz, statevector/shot simulator, COBYLA optimizer
├── recommender.py            # Inference engine (Research Mode & Demo Mode)
├── train.py                  # Model training entrypoint
├── app.py                    # Flask web application server & REST API
├── requirements.txt          # Package dependencies
│
├── experiments/              # Research Experiment Suite
│   ├── run_all.py            # Master automated experiment runner
│   ├── baselines.py          # Popularity, Content-Based, SVD baselines
│   ├── evaluate.py           # Ranking metrics (Precision@K, NDCG@K, HitRate@K, RMSE)
│   ├── ablation.py           # Qubit, Layer, and Entanglement ablation study
│   └── noise_experiment.py   # Shot variance & depolarizing noise sensitivity analysis
│
├── results/                  # Raw Empirical Outputs (Generated by experiments/run_all.py)
│   ├── metrics.csv           # Model metric comparisons
│   ├── circuit_metrics.csv   # Circuit depth & gate counts
│   ├── ablation.csv          # Ablation study output
│   ├── noise_experiment.csv  # Shot analysis output
│   └── experiment_config.json# Run metadata & seeds
│
├── paper/                    # Paper Support Assets
│   ├── IEEE_Paper_Draft.md   # IEEE-style research paper manuscript
│   ├── generate_paper_artifacts.py # Script generating LaTeX tables & figures from CSVs
│   ├── tables/               # Generated .tex and .md tables
│   └── figures/              # Generated publication-quality PNG figures
│
├── educational/              # Isolated Educational Algorithms
│   └── quantum_algorithms.py # Deutsch-Jozsa, Grover, Shor N=15, VQE, QGAN
│
├── docs/                     # Research Documentation
│   └── assumptions.md        # Explicit experimental assumptions & protocol specs
│
├── scripts/                  # Helper Utilities
│   └── check_environment.py  # Environment check script
│
└── tests/                    # Automated Test Suite
    └── test_project.py       # Pytest unit tests
```

---

## 6. How to Reproduce Experiments

### Step 1: Environment Audit
```bash
python scripts/check_environment.py
```

### Step 2: Run Unit Tests
```bash
pytest tests/test_project.py
```

### Step 3: Execute Master Experimental Pipeline
```bash
python experiments/run_all.py
```
*(Executes data downloading, temporal splitting, baseline fitting, VQC training, ranking evaluation, ablation study, and exports all CSV results to `results/`)*

### Step 4: Generate IEEE Paper Tables & Figures
```bash
python paper/generate_paper_artifacts.py
```

### Step 5: Launch Web Application
```bash
python app.py
```
*(Open http://127.0.0.1:5000 in your browser to interact with Research Mode and Demo Mode)*

---

## 7. Limitations & Honest Scientific Assessment
1. **Classical Preprocessing Bottleneck**: Features are compressed from 76D to 4D via linear PCA before quantum angle encoding. Non-linear feature interactions may be lost during classical dimensionality reduction.
2. **Simulator Execution**: Experiments were conducted using exact statevector and finite-shot quantum simulators (`Qiskit Aer`). Physical quantum hardware execution requires active IBM Quantum backend credentials.
3. **No Claim of Quantum Advantage**: The hybrid VQC model demonstrates feasibility and competitive performance against content-based baselines, but does **not** achieve exponential speedup or superior performance over tuned collaborative filtering matrix factorization.

---

## 8. License
This project is released under the [MIT License](LICENSE).
