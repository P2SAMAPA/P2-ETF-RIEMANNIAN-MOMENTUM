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
    if s >= 0:
        return f'<span class="score-positive">+{s:.4f}</span>'
    return f'<span class="score-negative">{s:.4f}</span>'

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
st.sidebar.markdown(f"- Momentum Lookback: **{config.MOMENTUM_LOOKBACK}** days")

st.markdown('<div class="main-header">🌀 P2Quant Riemannian Momentum</div>', unsafe_allow_html=True)
st.markdown('<div>Manifold Learning on SPD Covariance Matrices – Nonlinear Market Momentum</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = daily['top_picks'].get(key, [])
        universe_data = daily['universes'].get(key, {})
        scores = universe_data.get('scores', {})

        if top:
            pick = top[0]
            st.markdown(f"""
            <div class="hero-card">
                <h2>🌀 Top Manifold Momentum: {pick['ticker']}</h2>
                <p>Score: {score_badge(pick['score'])}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### All ETFs (Manifold Momentum Score)")
        rows = []
        for t, s in scores.items():
            rows.append({'Ticker': t, 'Score': f"{s:.4f}"})
        df = pd.DataFrame(rows).sort_values('Score', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
