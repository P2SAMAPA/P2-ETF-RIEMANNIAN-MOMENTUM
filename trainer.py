"""
Main training script – Daily, Global, and Shrinking‑Window modes.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from riemannian_model import RiemannianMomentum
import push_results


def compute_combined_scores(manifold_scores: dict, returns: pd.DataFrame,
                            tickers: list, lookback: int = 21) -> dict:
    """
    Combine manifold loading with 21‑day momentum return.
    Score = manifold_score * sign(momentum_return) * abs(momentum_return)
    """
    momentum = returns.iloc[-lookback:].mean().to_dict()
    combined = {}
    for t in tickers:
        ms = manifold_scores.get(t, 0.0)
        ret = momentum.get(t, 0.0)
        combined[t] = ms * np.sign(ret) * abs(ret)
    return combined


def run_mode(returns, tickers, model, mode_name, data_slice):
    """Run manifold momentum on a data slice and compute combined scores."""
    data = data_slice(returns)
    if len(data) < config.MIN_OBSERVATIONS:
        return None

    result = model.compute_manifold_momentum(data)
    manifold_scores = result['scores']
    confidence_intervals = result.get('confidence_intervals', {})

    if not manifold_scores:
        return None

    combined_scores = compute_combined_scores(manifold_scores, data, tickers, config.RETURN_LOOKBACK)

    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_picks = [
        {
            'ticker': t,
            'combined_score': s,
            'manifold_score': manifold_scores.get(t, 0.0),
            'ci_lower': confidence_intervals.get(t, {}).get('lower', manifold_scores.get(t, 0.0)),
            'ci_upper': confidence_intervals.get(t, {}).get('upper', manifold_scores.get(t, 0.0)),
        }
        for t, s in sorted_combined[:3]
    ]

    return {
        'mode': mode_name,
        'top_picks': top_picks,
        'all_combined_scores': combined_scores,
        'all_manifold_scores': manifold_scores,
        'confidence_intervals': confidence_intervals,
        'training_start': str(data.index[0].date()),
        'training_end': str(data.index[-1].date()),
        'n_observations': len(data)
    }


def run_shrinking_windows(df_master, tickers, model):
    """Fixed shrinking windows – use actual window data, NOT the same recent slice."""
    results = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp("2024-12-31")
        mask = (df_master['Date'] >= start_date) & (df_master['Date'] <= end_date)
        window_df = df_master[mask]
        returns_win = data_manager.prepare_returns_matrix(window_df, tickers)
        if len(returns_win) < config.MIN_OBSERVATIONS:
            continue

        result = model.compute_manifold_momentum(returns_win)
        manifold_scores = result['scores']
        if not manifold_scores:
            continue

        combined_scores = compute_combined_scores(manifold_scores, returns_win, tickers, config.RETURN_LOOKBACK)
        best_ticker = max(combined_scores, key=combined_scores.get)
        results.append({
            'window_start': start_year,
            'window_end': 2024,
            'ticker': best_ticker,
            'combined_score': combined_scores[best_ticker],
            'manifold_score': manifold_scores.get(best_ticker, 0.0)
        })

    if results:
        # Consensus
        vote = {}
        for r in results:
            t = r['ticker']
            vote[t] = vote.get(t, 0) + 1
        pick = max(vote, key=vote.get)
        conviction = vote[pick] / len(results) * 100
        return {
            'ticker': pick,
            'conviction': conviction,
            'num_windows': len(results),
            'num_pick_windows': vote[pick],
            'windows': results
        }
    return None


def run_riemannian():
    print(f"=== P2-ETF-RIEMANNIAN-MOMENTUM Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master['Date'] = pd.to_datetime(df_master['Date'])

    model = RiemannianMomentum(
        cov_window=config.COVARIANCE_WINDOW,
        frechet_window=config.FRECHET_WINDOW,
        momentum_lookbacks=config.MOMENTUM_LOOKBACKS,
        n_bootstrap=config.N_BOOTSTRAP,
        frechet_max_iter=config.FRECHET_MAX_ITER,
        frechet_tol=config.FRECHET_TOL
    )

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns_full = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_full) < config.MIN_OBSERVATIONS:
            continue

        universe_output = {}

        # Daily (504d)
        daily_out = run_mode(returns_full, tickers, model, "Daily (504d)",
                             lambda r: r.iloc[-config.DAILY_LOOKBACK:])
        if daily_out:
            universe_output['daily'] = daily_out
            print(f"  Daily top: {daily_out['top_picks'][0]['ticker']}")

        # Global (2008‑YTD)
        global_out = run_mode(returns_full, tickers, model, "Global (2008‑YTD)",
                              lambda r: r[r.index >= config.GLOBAL_TRAIN_START])
        if global_out:
            universe_output['global'] = global_out
            print(f"  Global top: {global_out['top_picks'][0]['ticker']}")

        # Shrinking windows consensus
        shrinking = run_shrinking_windows(df_master, tickers, model)
        if shrinking:
            universe_output['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_output

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "universes": all_results
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_riemannian()
