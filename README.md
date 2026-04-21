# P2-ETF-RIEMANNIAN-MOMENTUM

**Riemannian Geometry on the SPD Manifold – Nonlinear Market Momentum for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-RIEMANNIAN-MOMENTUM/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-RIEMANNIAN-MOMENTUM/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--riemannian--momentum--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-riemannian-momentum-results)

## Overview

`P2-ETF-RIEMANNIAN-MOMENTUM` uses Riemannian geometry on the space of symmetric positive definite (SPD) covariance matrices to capture nonlinear rotations in market structure. It computes a **manifold momentum** direction and ranks ETFs by their alignment with this direction.

## Methodology

1. **Rolling Covariance Estimation**: Compute SPD covariance matrices over a 63‑day window.
2. **Fréchet Mean**: Riemannian barycenter of recent covariances (21‑day baseline).
3. **Tangent Vector**: Logarithm map from baseline to current covariance, representing the direction of change on the manifold.
4. **Manifold Momentum Score**: Project recent ETF returns onto the dominant eigenvector of the tangent matrix.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)

## Usage
```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
