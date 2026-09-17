"""
Quantum Algorithms Module (Educational Shim).
Re-exports syllabus algorithms from the educational package.
"""

from educational.quantum_algorithms import (
    run_deutsch_jozsa,
    run_bernstein_vazirani,
    run_simon,
    run_grover,
    run_shor,
    run_vqe,
    run_qgan
)

__all__ = [
    "run_deutsch_jozsa",
    "run_bernstein_vazirani",
    "run_simon",
    "run_grover",
    "run_shor",
    "run_vqe",
    "run_qgan"
]
