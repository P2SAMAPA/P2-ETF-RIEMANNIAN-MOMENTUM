"""
Main training script for Riemannian Momentum engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from riemannian_model import RiemannianMomentum
import push_results

def run_riemannian():
    print(f"=== P2-ETF-RIEMANNIAN-MOMENTUM Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    model = RiemannianMomentum(
        cov_window=config.COVARIANCE_WINDOW,
        frechet_window=config.FRECHET_WINDOW,
        momentum_lookbacks=config.MOMENTUM_LOOKBACKS,
        n_bootstrap=config.N_BOOTSTRAP,
        frechet_max_iter=config.FRECHET_MAX_ITER,
        frechet_tol=config.FRECHET_TOL
    )

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        # Use recent data for manifold analysis (up to 4 years)
        recent_returns = returns.iloc[-min(len(returns), 1008):]
        result = model.compute_manifold_momentum(recent_returns)
        scores = result['scores']
        confidence_intervals = result.get('confidence_intervals', {})

        if not scores:
            continue

        all_results[universe_name] = {
            'tangent_vector': result['tangent_vector'],
            'scores': scores,
            'confidence_intervals': confidence_intervals
        }

        sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_picks[universe_name] = [
            {
                'ticker': t,
                'score': s,
                'ci_lower': confidence_intervals.get(t, {}).get('lower', s),
                'ci_upper': confidence_intervals.get(t, {}).get('upper', s)
            }
            for t, s in sorted_tickers[:3]
        ]
        print(f"  Top 3: {[p['ticker'] for p in top_picks[universe_name]]}")

    # Shrinking windows (simplified)
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue

        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            returns_win = data_manager.prepare_returns_matrix(df_window, tickers)
            if len(returns_win) < config.MIN_OBSERVATIONS:
                continue
            recent_win = returns_win.iloc[-min(len(returns_win), 1008):]
            result_win = model.compute_manifold_momentum(recent_win)
            scores_win = result_win['scores']
            if scores_win:
                best = max(scores_win, key=scores_win.get)
                window_top[universe_name] = {
                    'ticker': best,
                    'score': scores_win[best]
                }
        shrinking_results[window_label] = {
            'start_year': start_year,
            'top_picks': window_top
        }

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "covariance_window": config.COVARIANCE_WINDOW,
            "frechet_window": config.FRECHET_WINDOW,
            "momentum_lookbacks": config.MOMENTUM_LOOKBACKS,
            "n_bootstrap": config.N_BOOTSTRAP
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        },
        "shrinking_windows": shrinking_results
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_riemannian()
