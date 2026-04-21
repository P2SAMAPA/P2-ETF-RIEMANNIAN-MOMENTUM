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

# --- Manifold Parameters (Reduced for GitHub Actions) ---
COVARIANCE_WINDOW = 63                # 3 months (reduced from 126)
FRECHET_WINDOW = 21                   # 1 month baseline (reduced from 63)
FRECHET_MAX_ITER = 30                 # Fewer iterations (was 100)
FRECHET_TOL = 1e-6                    # Relaxed tolerance (was 1e-8)
MIN_OBSERVATIONS = 252                # Minimum data required
RANDOM_SEED = 42

# --- Momentum Parameters (Reduced) ---
MOMENTUM_LOOKBACKS = [5, 21]          # Only two horizons (was [5,10,21])
N_BOOTSTRAP = 10                      # Minimal bootstrap (was 50)

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
