"""
Quantum Model Core Module.

Contains 4-Qubit Angle Feature Map Encoding, Variational Quantum Circuit (VQC) ansatz,
statevector preference probability evaluation, and SciPy COBYLA optimizer.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from config import N_QUBITS, VQC_LAYERS


class AngleEncoding:
    """
    4-Qubit Angle Encoding Feature Map.
    Maps input features x in [0, pi]^4 to Ry rotations.
    """

    def __init__(self, n_qubits: int = N_QUBITS):
        self.n_qubits = n_qubits
        self.x_params = ParameterVector("x", n_qubits)

    def build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, name="AngleEncoding")
        for i in range(self.n_qubits):
            qc.ry(self.x_params[i], i)
        return qc


class VariationalQuantumCircuit:
    """
    4-Qubit Variational Quantum Circuit (VQC) Engine.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = VQC_LAYERS):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = AngleEncoding(n_qubits=n_qubits)
        self.num_params = (n_layers + 1) * n_qubits * 2
        self.theta_params = ParameterVector("theta", self.num_params)

    def build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, name="VQC_Ansatz")
        idx = 0
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qc.ry(self.theta_params[idx], i)
                idx += 1
                qc.rz(self.theta_params[idx], i)
                idx += 1
            for i in range(self.n_qubits):
                qc.cx(i, (i + 1) % self.n_qubits)
                
        for i in range(self.n_qubits):
            qc.ry(self.theta_params[idx], i)
            idx += 1
            qc.rz(self.theta_params[idx], i)
            idx += 1
        return qc

    def get_full_circuit(self) -> QuantumCircuit:
        fm = self.feature_map.build_circuit()
        ansatz = self.build_ansatz()
        qc = QuantumCircuit(self.n_qubits, name="Full_VQC")
        qc.compose(fm, inplace=True)
        qc.compose(ansatz, inplace=True)
        return qc

    def evaluate_preference(self, x_features: np.ndarray, theta_values: np.ndarray, target_qubit: int = 0) -> float:
        """
        Evaluates exact statevector probability of measuring |1> on target_qubit.
        Returns continuous preference prediction score in range [0.0, 1.0].
        """
        full_qc = self.get_full_circuit()
        param_dict = {}
        for i in range(self.n_qubits):
            param_dict[self.feature_map.x_params[i]] = x_features[i]
        for i in range(self.num_params):
            param_dict[self.theta_params[i]] = theta_values[i]
            
        bound_qc = full_qc.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound_qc)
        probs = sv.probabilities()
        
        prob_one = sum(prob for state_idx, prob in enumerate(probs) if (state_idx >> target_qubit) & 1)
        return float(prob_one)

    def evaluate_batch(self, X_features: np.ndarray, theta_values: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate_preference(x, theta_values) for x in X_features])


class VQCOptimizer:
    """
    Classical SciPy COBYLA Optimizer wrapper for VQC parameters.
    """

    def __init__(self, vqc: VariationalQuantumCircuit, max_iter: int = 25):
        self.vqc = vqc
        self.max_iter = max_iter

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        loss_history = []
        
        def objective(theta):
            preds = self.vqc.evaluate_batch(X_train, theta)
            mse_loss = float(np.mean((preds - y_train) ** 2))
            loss_history.append(mse_loss)
            return mse_loss

        np.random.seed(42)
        initial_theta = np.random.uniform(0, 2 * np.pi, self.vqc.num_params)
        
        if verbose:
            print(f"Starting VQC Optimization (COBYLA max_iter={self.max_iter})...")

        res = minimize(objective, x0=initial_theta, method="COBYLA", options={"maxiter": self.max_iter})
        
        if verbose:
            print(f"VQC Optimization finished. Final Loss: {res.fun:.4f}")

        return res.x, loss_history
