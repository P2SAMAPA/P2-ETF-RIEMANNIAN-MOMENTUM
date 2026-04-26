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

# --- Macro Columns ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]   # available from 2008

# --- Manifold Parameters ---
COVARIANCE_WINDOW = 126
FRECHET_WINDOW = 21
FRECHET_MAX_ITER = 42
FRECHET_TOL = 1e-6
MIN_OBSERVATIONS = 252
RANDOM_SEED = 42

# --- Momentum Parameters ---
MOMENTUM_LOOKBACKS = [5, 21]
N_BOOTSTRAP = 30

# --- Signal Parameters ---
RETURN_LOOKBACK = 21              # days for momentum return (used for predictive score)

# --- Training Modes ---
DAILY_LOOKBACK = 504              # days for daily training
GLOBAL_TRAIN_START = "2008-01-01"

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
