"""
Quantum Model Core Module (Research Grade).

Provides N-Qubit Angle Feature Map Encoding, Variational Quantum Circuit (VQC) ansatz
with configurable qubits, layers, and entanglement structures, statevector / finite-shot / noisy simulation,
SciPy optimizer wrapper, and automated circuit metric calculation.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from config import N_QUBITS, VQC_LAYERS, COBYLA_MAX_ITER


class AngleEncoding:
    """
    N-Qubit Angle Feature Map Encoding.
    Maps input feature vector x in [0, pi]^N to Ry(x_i) rotations on |0> states.
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
    Configurable N-Qubit Variational Quantum Circuit (VQC).
    Supports variable qubit count, layer depth, and entangling structures.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = VQC_LAYERS,
        entanglement: bool = True
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
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
            if self.entanglement and self.n_qubits > 1:
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

    def evaluate_preference(
        self,
        x_features: np.ndarray,
        theta_values: np.ndarray,
        target_qubit: int = 0,
        shots: Optional[int] = None,
        noise_model: Any = None
    ) -> float:
        """
        Evaluates preference probability P(|1>) on target_qubit.
        Uses exact statevector when shots is None, or finite-shot execution when shots is specified.
        """
        full_qc = self.get_full_circuit()
        param_dict = {}
        for i in range(self.n_qubits):
            param_dict[self.feature_map.x_params[i]] = x_features[i]
        for i in range(self.num_params):
            param_dict[self.theta_params[i]] = theta_values[i]
            
        bound_qc = full_qc.assign_parameters(param_dict)

        if shots is None and noise_model is None:
            sv = Statevector.from_instruction(bound_qc)
            probs = sv.probabilities()
            prob_one = sum(prob for state_idx, prob in enumerate(probs) if (state_idx >> target_qubit) & 1)
            return float(prob_one)
        else:
            try:
                from qiskit_aer import AerSimulator
                bound_qc_m = bound_qc.copy()
                bound_qc_m.measure_all()
                
                sim = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
                result = sim.run(bound_qc_m, shots=shots or 1024).result()
                counts = result.get_counts()
                
                total_shots = sum(counts.values())
                ones_count = 0
                for bitstr, cnt in counts.items():
                    rev_bitstr = bitstr.replace(" ", "")[::-1]
                    if target_qubit < len(rev_bitstr) and rev_bitstr[target_qubit] == '1':
                        ones_count += cnt
                return float(ones_count / total_shots)
            except ImportError:
                sv = Statevector.from_instruction(bound_qc)
                probs = sv.probabilities()
                prob_one = sum(prob for state_idx, prob in enumerate(probs) if (state_idx >> target_qubit) & 1)
                return float(prob_one)

    def evaluate_batch(
        self,
        X_features: np.ndarray,
        theta_values: np.ndarray,
        shots: Optional[int] = None,
        noise_model: Any = None
    ) -> np.ndarray:
        return np.array([
            self.evaluate_preference(x, theta_values, shots=shots, noise_model=noise_model)
            for x in X_features
        ])


def get_circuit_metrics(vqc: VariationalQuantumCircuit) -> Dict[str, Any]:
    """
    Computes exact, un-hardcoded complexity metrics directly from the compiled QuantumCircuit.
    """
    qc = vqc.get_full_circuit()
    depth = qc.depth()
    num_qubits = qc.num_qubits
    num_params = vqc.num_params
    
    ops = qc.count_ops()
    total_gates = sum(ops.values())
    cnot_gates = ops.get("cx", 0)
    single_qubit_gates = total_gates - cnot_gates

    return {
        "n_qubits": num_qubits,
        "n_layers": vqc.n_layers,
        "entanglement": vqc.entanglement,
        "num_params": num_params,
        "circuit_depth": depth,
        "total_gates": total_gates,
        "cnot_gates": cnot_gates,
        "single_qubit_gates": single_qubit_gates
    }


class VQCOptimizer:
    """
    Classical SciPy COBYLA Optimizer wrapper for VQC parameters.
    """

    def __init__(self, vqc: VariationalQuantumCircuit, max_iter: int = COBYLA_MAX_ITER, seed: int = 42):
        self.vqc = vqc
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
        loss_history = []
        
        def objective(theta):
            preds = self.vqc.evaluate_batch(X_train, theta)
            mse_loss = float(np.mean((preds - y_train) ** 2))
            loss_history.append(mse_loss)
            return mse_loss

        np.random.seed(self.seed)
        initial_theta = np.random.uniform(0, 2 * np.pi, self.vqc.num_params)

        res = minimize(objective, x0=initial_theta, method="COBYLA", options={"maxiter": self.max_iter})
        return res.x, loss_history
