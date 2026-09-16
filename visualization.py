"""
Visualization Module.

Provides plotting helpers for circuit rendering, user genre profiles, training loss,
recommendation scores, VQE convergence, and QGAN distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from config import ALL_GENRES


def draw_circuit(circuit: QuantumCircuit) -> plt.Figure:
    """Draws a Qiskit circuit using Matplotlib or fallback ASCII text."""
    try:
        fig = circuit.draw(output="mpl", fold=80)
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.05, 0.5, str(circuit.draw(output="text", fold=70)), fontfamily="monospace", fontsize=9, va="center")
        ax.axis("off")
        return fig


def plot_user_profile(user_profile: np.ndarray, user_id: int = 1) -> plt.Figure:
    """Bar chart of user genre preference vector."""
    fig, ax = plt.subplots(figsize=(9, 4))
    genres = ALL_GENRES[:len(user_profile)]
    colors = ["#4C72B0" if v >= 0.5 else "#C44E52" for v in user_profile]
    
    ax.bar(genres, user_profile, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Neutral Threshold (0.5)")
    ax.set_ylabel("Preference Weight")
    ax.set_title(f"User #{user_id} Genre Preference Profile", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_loss_history(loss_history: list) -> plt.Figure:
    """Line plot of training loss."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", color="#1F77B4", linewidth=2)
    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title("VQC Training Loss Curve", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_recommendation_scores(ranked_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of Top-K recommendation scores."""
    fig, ax = plt.subplots(figsize=(9, 4))
    df = ranked_df.head(5).iloc[::-1]
    titles = [t[:25] + "..." if len(t) > 25 else t for t in df["title"]]
    scores = df["quantum_score"]
    
    bars = ax.barh(titles, scores, color="#2CA02C", edgecolor="black", alpha=0.85)
    ax.axvline(0.5, color="gray", linestyle="--", label="Preference Threshold (0.5)")
    ax.set_xlabel("Quantum Preference Score P(|1⟩)")
    ax.set_title("Top-5 Recommended Movie Preference Scores", fontweight="bold")
    ax.set_xlim(0.0, 1.05)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_vqe(energy_history: list, exact_energy: float) -> plt.Figure:
    """Plots VQE convergence."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(1, len(energy_history) + 1), energy_history, marker="o", color="#9467BD", label="VQE Estimated Energy")
    ax.axhline(exact_energy, color="#D62728", linestyle="--", label=f"Exact Ground Energy ({exact_energy:.4f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("VQE Energy Convergence", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_qgan(real_dist: np.ndarray, synthetic_dist: np.ndarray) -> plt.Figure:
    """Plots QGAN real vs synthetic distributions."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    labels = [f"|{bin(i)[2:].zfill(2)}>" for i in range(len(real_dist))]
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, real_dist, width, label="Real Distribution", color="#1F77B4")
    ax.bar(x + width/2, synthetic_dist, width, label="QGAN Synthetic", color="#FF7F0E")
    ax.set_ylabel("Probability")
    ax.set_title("QGAN Preference Profile Generator Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig
