"""Full benchmark dashboard with Plotly chart and downloadable report."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st

from app.engine_factory import get_engine
from src.config import config

st.set_page_config(page_title="Benchmark Dashboard", page_icon="📊", layout="wide")

st.title("📊 Benchmark Dashboard")
st.caption("Strategy A vs Strategy B across all benchmark queries")

engine = get_engine()

BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "What happens when the database goes down?",
    "How do you keep users logged in?",
    "How do you know when something breaks?",
    "What if an entire region fails?",
]


@st.cache_data(show_spinner="Running all benchmark queries…")
def run_all_benchmarks(top_k: int) -> list[dict]:
    results = []
    for query in BENCHMARK_QUERIES:
        a_results = engine.strategy_a_search(query, k=top_k)
        expanded_query, b_results = engine.strategy_b_search(query, k=top_k)
        avg_a = statistics.mean(r.score for r in a_results) if a_results else 0.0
        avg_b = statistics.mean(r.score for r in b_results) if b_results else 0.0
        results.append({
            "query": query,
            "expanded_query": expanded_query,
            "strategy_a": a_results,
            "strategy_b": b_results,
            "avg_a": avg_a,
            "avg_b": avg_b,
            "delta": avg_b - avg_a,
        })
    return results


results = run_all_benchmarks(config.top_k)

# ── Summary metrics ────────────────────────────────────────────────────────────
overall_avg_a = statistics.mean(e["avg_a"] for e in results)
overall_avg_b = statistics.mean(e["avg_b"] for e in results)
overall_delta = overall_avg_b - overall_avg_a
wins_b = sum(1 for e in results if e["delta"] > 0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Overall Avg — Strategy A", f"{overall_avg_a:.4f}")
m2.metric("Overall Avg — Strategy B", f"{overall_avg_b:.4f}", delta=f"{overall_delta:+.4f}")
m3.metric("Strategy B Wins", f"{wins_b}/{len(results)}")
m4.metric("Corpus", "100 GCP chunks")

st.divider()

# ── Grouped bar chart ─────────────────────────────────────────────────────────
query_labels = [f"Q{i+1}" for i in range(len(results))]
fig = go.Figure(data=[
    go.Bar(
        name="Strategy A — Raw",
        x=query_labels,
        y=[e["avg_a"] for e in results],
        marker_color="#60a5fa",
        text=[f"{e['avg_a']:.4f}" for e in results],
        textposition="outside",
    ),
    go.Bar(
        name="Strategy B — AI-Enhanced",
        x=query_labels,
        y=[e["avg_b"] for e in results],
        marker_color="#4ade80",
        text=[f"{e['avg_b']:.4f}" for e in results],
        textposition="outside",
    ),
])
fig.update_layout(
    barmode="group",
    title="Average Cosine Similarity Score per Query",
    yaxis=dict(title="Avg Cosine Score", range=[0, 1.05]),
    xaxis_title="Query",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    plot_bgcolor="white",
    paper_bgcolor="white",
)
for i, entry in enumerate(results):
    delta_label = f"+{entry['delta']:.4f}" if entry["delta"] > 0 else f"{entry['delta']:.4f}"
    fig.add_annotation(
        x=query_labels[i], y=max(entry["avg_a"], entry["avg_b"]) + 0.04,
        text=f"Δ {delta_label}", showarrow=False,
        font=dict(size=11, color="#15803d" if entry["delta"] > 0 else "#dc2626"),
    )

st.plotly_chart(fig, use_container_width=True)

# Query label legend
with st.expander("Query labels"):
    for i, e in enumerate(results):
        st.markdown(f"**Q{i+1}:** {e['query']}")

st.divider()

# ── Per-query detail ──────────────────────────────────────────────────────────
st.subheader("Per-Query Results")

for i, entry in enumerate(results):
    with st.expander(f"Q{i+1}: {entry['query']}  — Δ avg score {entry['delta']:+.4f}"):
        st.markdown(f"**Strategy B Expanded Query:**  \n> *{entry['expanded_query']}*")
        st.markdown("")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Strategy A — Raw**")
            for r in entry["strategy_a"]:
                st.markdown(f"`#{r.rank}` · score `{r.score:.4f}`")
                st.caption(r.text[:200] + "…")
        with col_b:
            st.markdown("**Strategy B — AI-Enhanced**")
            for r in entry["strategy_b"]:
                st.markdown(f"`#{r.rank}` · score `{r.score:.4f}`")
                st.caption(r.text[:200] + "…")

        st.metric(
            "Score delta (B − A)",
            f"{entry['delta']:+.4f}",
            delta=f"{'Strategy B wins' if entry['delta'] > 0 else 'Strategy A wins'}",
        )

st.divider()

# ── FAISS Index Explorer ──────────────────────────────────────────────────────
st.subheader("FAISS Index Explorer")
st.caption("Inspect what is stored in the in-memory vector index")

import json
store = engine._store
corpus = json.loads(config.data_path.read_text(encoding="utf-8"))

col_i1, col_i2, col_i3 = st.columns(3)
col_i1.metric("Vectors in index", store._index.ntotal)
col_i2.metric("Vector dimensions", store._dim)
col_i3.metric("Index type", "IndexFlatIP (Cosine)")

st.markdown("**All indexed chunks** — every row is one embedded document:")
rows = []
for chunk_id, text in store._id_to_chunk.items():
    topic = corpus[chunk_id]["topic"] if chunk_id < len(corpus) else "unknown"
    rows.append({"chunk_id": chunk_id, "topic": topic, "text_preview": text[:120] + "…"})

import pandas as pd
st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)

st.divider()

# ── Download benchmark report ─────────────────────────────────────────────────
if config.benchmark_md_path.exists():
    st.download_button(
        label="⬇  Download retrieval_benchmark.md",
        data=config.benchmark_md_path.read_text(encoding="utf-8"),
        file_name="retrieval_benchmark.md",
        mime="text/markdown",
    )
else:
    st.warning("Run `python run_benchmark.py` first to generate `retrieval_benchmark.md`.")
