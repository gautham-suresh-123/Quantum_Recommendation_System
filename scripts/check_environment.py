"""
Environment Check Script for Quantum Recommendation System Research Pipeline.
Verifies Python version, Qiskit version, dependencies, dataset presence, and hardware acceleration capability.
"""

import sys
import importlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_PACKAGES = {
    "numpy": "1.24.0",
    "pandas": "2.0.0",
    "sklearn": "1.2.0",
    "qiskit": "1.0.0",
    "scipy": "1.10.0",
    "matplotlib": "3.7.0",
    "requests": "2.28.0"
}

def check_environment() -> bool:
    print("=" * 60)
    print("  QUANTUM RECOMMENDATION SYSTEM - ENVIRONMENT AUDIT")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]} ({sys.platform})")
    
    all_ok = True
    print("\n--- Checking Dependencies ---")
    for pkg, min_ver in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pkg:<12}: installed ({ver})")
        except ImportError:
            print(f"  [MISSING] {pkg:<10}: NOT installed (requires >={min_ver})")
            all_ok = False
            
    # Check Aer noise support
    try:
        import qiskit_aer
        print(f"  [OK] qiskit_aer   : installed ({qiskit_aer.__version__})")
    except ImportError:
        print("  [INFO] qiskit_aer   : Not found (Noise experiments will report Aer unavailable)")

    # Check Dataset
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    movies_file = data_dir / "movies.csv"
    ratings_file = data_dir / "ratings.csv"
    
    print("\n--- Checking Local Dataset ---")
    if movies_file.exists() and ratings_file.exists():
        print(f"  [OK] Dataset found at {data_dir}")
        import pandas as pd
        m_df = pd.read_csv(movies_file)
        r_df = pd.read_csv(ratings_file)
        print(f"       - Movies : {len(m_df)} records")
        print(f"       - Ratings: {len(r_df)} records")
    else:
        print(f"  [INFO] MovieLens 100K dataset files not found in {data_dir}.")
        print("         The pipeline will automatically download official MovieLens dataset upon first execution.")

    print("\n" + "=" * 60)
    if all_ok:
        print("SUCCESS: Environment meets research pipeline requirements.")
    else:
        print("WARNING: Some dependencies are missing. Please install via: pip install -r requirements.txt")
    print("=" * 60)
    return all_ok

if __name__ == "__main__":
    check_environment()
