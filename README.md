# Quantum-Assisted Personalized Recommendation System Using Variational Quantum Circuits

> **B.Tech Mini-Project in Artificial Intelligence & Data Science**  
> *Simplified, Viva-Friendly Real Working Implementation*

---

## 1. Abstract
This project implements a personalized movie recommendation system powered by a **4-Qubit Variational Quantum Circuit (VQC)** using the MovieLens 100K dataset. User preference vectors and movie genre features are combined, reduced to 4 dimensions using PCA, and mapped onto 4 qubits via Angle Encoding ($R_y$). A 2-layer parameterized quantum circuit measures statevector probabilities $P(|1\rangle) \in [0.0, 1.0]$ to generate preference scores and rank candidate movies. An interactive **Streamlit** application provides recommendation outputs alongside an educational **Quantum Algorithm Lab** demonstrating syllabus algorithms (Deutsch-Jozsa, Bernstein-Vazirani, Simon, Grover, Shor $N=15$, VQE, QGAN).

---

## 2. Project Structure

```text
quantum_recommender/
│
├── app.py                  # Flask Web App Server (HTML5/CSS3/JS SPA with 5 Tab Views & REST API)
├── templates/              # HTML5 single-page application template (index.html)
├── static/                 # Vanilla CSS glassmorphism styles (style.css) & JS client (app.js)
├── train.py                # Model training script (supports DEMO_MODE = True for fast viva run)
├── config.py               # Hyperparameters & settings
├── requirements.txt        # Package requirements
├── README.md               # Documentation
│
├── data/                   # MovieLens 100K CSV files
├── models/                 # Model artifacts (vqc_recommender.pkl)
│
├── preprocessing.py        # Data loading, multi-hot genre encoding, user profile, PCA
├── quantum_model.py        # 4-Qubit Angle Encoding, 2-Layer VQC, COBYLA optimizer
├── recommender.py          # Candidate selection, VQC scoring, Top-K ranking
├── quantum_algorithms.py   # Deutsch-Jozsa, Bernstein-Vazirani, Simon, Grover, Shor N=15, VQE, QGAN
├── visualization.py        # Matplotlib charts & circuit drawers
│
└── tests/
    └── test_project.py     # All-in-one test suite
```

---

## 3. Simplified Architecture

```text
MovieLens 100K ──► User Profile (19D) + Movie Features (19D)
                       │
                       ▼
            Combined Feature Pair (38D)
                       │
                       ▼
               PCA Reduction (4D)
                       │
                       ▼
            4-Qubit Angle Encoding [Ry(x_i)]
                       │
                       ▼
          4-Qubit VQC (2 Layers + CNOT Rings)
                       │
                       ▼
         Quantum Preference Score P(|1⟩)
                       │
                       ▼
          Top-5 Personalized Recommendations
```

---

## 4. How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train VQC Model Offline
```bash
python train.py
```
*(By default `DEMO_MODE = True` in `config.py` runs fast 2-3 second training for viva demonstration)*

### 3. Launch HTML5/CSS/JS Web Application (Flask)
```bash
python app.py
```
*(Open http://127.0.0.1:5000 in your web browser)*

### 4. Run Automated Unit Tests
```bash
pytest tests/test_project.py
```

---

## 5. Viva Explanation Guide (2-3 Minutes)

**Question: "How does your quantum recommendation system work?"**

> *"We create a user preference vector from historical ratings and represent candidate movies using genre multi-hot vectors. We combine these features and reduce them to 4 dimensions using PCA. These 4 features are encoded into 4 qubits using Ry rotation gates. A 2-layer Variational Quantum Circuit (VQC) with Ry/Rz rotations and entangling CNOT gates processes the quantum state and measures statevector probability P(|1⟩) as a preference prediction score. Candidate movies are ranked by score to produce Top-5 recommendations."*

---

## 6. Group Member Work Division (4 Students)
- **Student 1**: Data preprocessing, genre multi-hot encoding, user profiles (`preprocessing.py`).
- **Student 2**: Quantum feature encoding, 4-qubit VQC ansatz design, model training (`quantum_model.py`, `train.py`).
- **Student 3**: Recommendation candidate selection, VQC preference scoring, ranking (`recommender.py`).
- **Student 4**: Streamlit application UI, visualizations, quantum algorithm lab integration (`app.py`, `quantum_algorithms.py`, `visualization.py`).
