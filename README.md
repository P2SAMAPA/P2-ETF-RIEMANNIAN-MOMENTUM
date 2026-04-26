# P2-ETF-RIEMANNIAN-MOMENTUM

**Riemannian Geometry on the SPD Manifold – Predictive Manifold Momentum for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-RIEMANNIAN-MOMENTUM/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-RIEMANNIAN-MOMENTUM/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--riemannian--momentum--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-riemannian-momentum-results)

## Overview

`P2-ETF-RIEMANNIAN-MOMENTUM` computes the **manifold momentum** on the space of symmetric positive definite (SPD) covariance matrices to capture nonlinear rotations in market structure. It then combines the geometric manifold loading with recent **momentum returns** to produce a predictive combined score, ranking ETFs by expected return across three training modes.

## Methodology

1. **Rolling SPD Covariance Estimation** – 126‑day rolling covariance matrices.
2. **Fréchet Mean** – Riemannian barycenter of baseline covariances (21‑day window).
3. **Tangent Vector** – Logarithm map from the Fréchet mean to the current covariance, representing the direction of structural change on the manifold.
4. **Manifold Loading** – Each ETF's loading on the **dominant eigenvector** of the tangent matrix (geometric contribution to deformation).
5. **Forward‑Looking Combined Score** – `Score = Manifold Loading × sign(21‑day Return) × |21‑day Return|`. This converts a purely geometric alignment into a predictive signal.
6. **Bootstrap Confidence Intervals** – 30 resamples provide 95% CI for the manifold loading.

## Training Modes

- **Daily (504d)** – Most recent 2 years for current regime awareness.
- **Global (2008‑YTD)** – Full history for long‑term structural relationships.
- **Shrinking Windows Consensus** – 15 rolling windows (2010‑2024), consensus ETF across windows with conviction scoring.

## Universe

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Macro Conditioning

Current macro values (VIX, DXY, T10Y2Y, TBILL_3M) are included in the node features so manifold computations are conditioned on the macro environment.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
Dashboard
Three sub‑tabs per universe: Daily, Global, Shrinking Consensus.

Hero card with combined score, manifold loading, and 95% CI.

Full ETF ranking tables.

Fixed Shrinking Windows
Each window uses actual data from the period (e.g., 2010‑01‑01 to 2024‑12‑31), not the same recent slice, ensuring distinct historical results.
