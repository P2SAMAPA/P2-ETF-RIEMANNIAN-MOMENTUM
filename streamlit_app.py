"""
Streamlit Dashboard for Riemannian Momentum Engine.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Riemannian Momentum", page_icon="🌀", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; }
    .score-positive { color: #28a745; font-weight: 600; }
    .score-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def score_badge(s):
    try:
        s = float(s)
    except:
        return f'{s}'
    if s >= 0:
        return f'<span class="score-positive">+{s:.4f}</span>'
    return f'<span class="score-negative">{s:.4f}</span>'

def render_mode_tab(mode_data, mode_name):
    if not mode_data:
        st.warning(f"No {mode_name} data available.")
        return
    top_picks = mode_data.get('top_picks', [])
    if not top_picks:
        st.info(f"No predictions for {mode_name}.")
        return
    pick = top_picks[0]
    ticker = pick['ticker']
    combined = pick.get('combined_score', 0.0)
    manifold = pick.get('manifold_score', 0.0)
    ci_lower = pick.get('ci_lower', manifold)
    ci_upper = pick.get('ci_upper', manifold)

    st.markdown(f"""
    <div class="hero-card">
        <h2>🌀 {mode_name} Top Pick</h2>
        <h1>{ticker}</h1>
        <p>Combined Score: {score_badge(combined)}</p>
        <p>Manifold Score: {manifold:.4f} (95% CI: {ci_lower:.4f} – {ci_upper:.4f})</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Top 3 Picks")
    rows = []
    for p in top_picks:
        rows.append({
            'Ticker': p['ticker'],
            'Combined Score': f"{p.get('combined_score',0):.4f}",
            'Manifold Score': f"{p.get('manifold_score',0):.4f}",
            'CI Lower': f"{p.get('ci_lower',0):.4f}",
            'CI Upper': f"{p.get('ci_upper',0):.4f}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # All ETFs table (from all_combined_scores)
    all_combined = mode_data.get('all_combined_scores', {})
    all_manifold = mode_data.get('all_manifold_scores', {})
    if all_combined:
        all_rows = []
        for t, cs in all_combined.items():
            ms = all_manifold.get(t, 0.0)
            all_rows.append({
                'Ticker': t,
                'Combined Score': f"{cs:.4f}",
                'Manifold Score': f"{ms:.4f}"
            })
        df_all = pd.DataFrame(all_rows).sort_values('Combined Score', ascending=False)
        st.markdown("### All ETFs")
        st.dataframe(df_all, use_container_width=True, hide_index=True)

def render_shrinking_tab(shrinking_data):
    if not shrinking_data:
        st.warning("No shrinking data.")
        return
    st.markdown(f"""
    <div class="hero-card">
        <h2>🔄 Shrinking Consensus</h2>
        <h1>{shrinking_data['ticker']}</h1>
        <p>{shrinking_data['conviction']:.0f}% conviction · {shrinking_data['num_windows']} windows</p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 All Windows"):
        rows = []
        for w in shrinking_data.get('windows', []):
            rows.append({
                'Window': f"{w['window_start']}-{w['window_end']}",
                'ETF': w['ticker'],
                'Combined Score': f"{w.get('combined_score',0):.4f}",
                'Manifold Score': f"{w.get('manifold_score',0):.4f}"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
st.sidebar.divider()
st.sidebar.markdown("### 🌀 Manifold Parameters")
st.sidebar.markdown(f"- Cov Window: **{config.COVARIANCE_WINDOW}** days")
st.sidebar.markdown(f"- Fréchet Window: **{config.FRECHET_WINDOW}** days")
st.sidebar.markdown(f"- Momentum Lookbacks: **{config.MOMENTUM_LOOKBACKS}**")
st.sidebar.markdown(f"- Bootstrap Samples: **{config.N_BOOTSTRAP}**")

st.markdown('<div class="main-header">🌀 P2Quant Riemannian Momentum</div>', unsafe_allow_html=True)
st.markdown('<div>Manifold Learning on SPD Covariance Matrices – Nonlinear Market Momentum</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

universes_data = data.get('universes', {})
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, keys):
    uni = universes_data.get(key, {})
    if not uni:
        with tab:
            st.info(f"No data for {key}.")
        continue
    with tab:
        d, g, s = st.tabs(["📅 Daily (504d)", "🌍 Global (2008‑YTD)", "🔄 Shrinking Consensus"])
        with d:
            render_mode_tab(uni.get('daily'), "Daily")
        with g:
            render_mode_tab(uni.get('global'), "Global")
        with s:
            render_shrinking_tab(uni.get('shrinking'))
