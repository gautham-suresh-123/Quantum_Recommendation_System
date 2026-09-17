import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from quantum_model import VariationalQuantumCircuit, AngleEncoding
from config import N_QUBITS, VQC_LAYERS, ALL_GENRES

def test_vqc_scratch():
    vqc = VariationalQuantumCircuit(n_qubits=N_QUBITS, n_layers=VQC_LAYERS)
    assert vqc.n_qubits == N_QUBITS
