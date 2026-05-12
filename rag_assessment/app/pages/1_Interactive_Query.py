"""Live side-by-side Strategy A vs Strategy B comparison."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.engine_factory import get_engine
from src.config import config

st.set_page_config(page_title="Interactive Query", page_icon="🔎", layout="wide")

st.markdown("""
<style>
    .result-card {
        background: #fafafa;
        border-left: 4px solid #6366f1;
        border-radius: 4px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88em;
        line-height: 1.5;
    }
    .result-card-b { border-left-color: #22c55e; }
    .score-bar-label { font-size: 0.78em; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

st.title("🔎 Interactive Query")
st.caption("Compare Strategy A (raw) vs Strategy B (AI-enhanced) in real time")

engine = get_engine()

# ── Controls ──────────────────────────────────────────────────────────────────
with st.form("query_form"):
    query = st.text_input(
        "Enter your query",
        value="How does the system handle peak load?",
        placeholder="Ask something about the cloud architecture…",
    )
    col_k, col_submit = st.columns([1, 4])
    with col_k:
        k = st.slider("Top-K results", min_value=1, max_value=5, value=config.top_k)
    submitted = col_submit.form_submit_button("▶  Run Both Strategies", use_container_width=True)

if not submitted:
    st.info("Enter a query above and click **Run Both Strategies** to begin.")
    st.stop()

# ── Execute ───────────────────────────────────────────────────────────────────
with st.spinner("Embedding and searching…"):
    a_results = engine.strategy_a_search(query, k=k)
    expanded_query, b_results = engine.strategy_b_search(query, k=k)

avg_a = sum(r.score for r in a_results) / len(a_results) if a_results else 0.0
avg_b = sum(r.score for r in b_results) / len(b_results) if b_results else 0.0
delta = avg_b - avg_a

# ── Expanded query callout ────────────────────────────────────────────────────
st.divider()
st.markdown("**Strategy B — Expanded Query:**")
st.info(expanded_query)

# ── Side-by-side results ──────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Strategy A — Raw Vector Search")
    for r in a_results:
        score_pct = int(r.score * 100)
        st.progress(min(score_pct, 100), text=f"Rank #{r.rank} · Cosine {r.score:.4f}")
        st.markdown(
            f'<div class="result-card">{r.text}</div>',
            unsafe_allow_html=True,
        )

with col_b:
    st.markdown("### Strategy B — AI-Enhanced Retrieval")
    for r in b_results:
        score_pct = int(r.score * 100)
        st.progress(min(score_pct, 100), text=f"Rank #{r.rank} · Cosine {r.score:.4f}")
        st.markdown(
            f'<div class="result-card result-card-b">{r.text}</div>',
            unsafe_allow_html=True,
        )

# ── Score comparison metrics ──────────────────────────────────────────────────
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Avg Score — Strategy A", f"{avg_a:.4f}")
m2.metric("Avg Score — Strategy B", f"{avg_b:.4f}", delta=f"{delta:+.4f}")
m3.metric("Winner", "Strategy B ✓" if delta > 0 else "Strategy A ✓")
