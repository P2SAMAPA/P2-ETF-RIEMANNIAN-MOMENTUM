"""
Configuration for P2-ETF-RIEMANNIAN-MOMENTUM engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-riemannian-momentum-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Manifold Parameters (Increased Complexity) ---
COVARIANCE_WINDOW = 126               # 6 months (was 63)
FRECHET_WINDOW = 63                   # 3 months baseline (was 21)
FRECHET_MAX_ITER = 100                # More iterations for convergence
FRECHET_TOL = 1e-8                    # Tighter tolerance
MIN_OBSERVATIONS = 504                # Minimum data required
RANDOM_SEED = 42

# --- Momentum Parameters ---
MOMENTUM_LOOKBACKS = [5, 10, 21]      # Multiple horizons to average
N_BOOTSTRAP = 50                      # Bootstrap samples for confidence intervals

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
