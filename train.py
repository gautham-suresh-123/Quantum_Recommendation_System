"""
Model Training Script (Research Grade).

Executes offline VQC training using leakage-free temporal splitting and canonical 76D feature extraction.
Fits Scaler, PCA, and VQC parameters strictly on the training split and persists the trained artifact to `models/vqc_recommender.pkl`.
"""

import sys
import logging
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.run_all import run_full_experiment_pipeline

def main():
    print("Executing offline model training & full experimental pipeline...")
    results = run_full_experiment_pipeline()
    print("Training and experimental evaluation completed successfully.")
    return results

if __name__ == "__main__":
    main()
