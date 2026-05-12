"""Landing page — project overview and pipeline architecture."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.engine_factory import get_engine
from src.config import config

st.set_page_config(
    page_title="Context-Aware RAG Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border: 1px solid #c7d2fe;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .strategy-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .tag-a { background: #dbeafe; color: #1d4ed8; }
    .tag-b { background: #dcfce7; color: #15803d; }
</style>
""", unsafe_allow_html=True)

# Warm up the engine on the landing page so other pages load instantly
with st.spinner("Loading RAG engine…"):
    engine = get_engine()

st.title("🔍 Context-Aware Retrieval Engine")
st.caption("Senior Gen AI Assessment — Semantic RAG & Vector Search")

st.markdown("""
A production-grade **Retrieval-Augmented Generation (RAG)** pipeline comparing two retrieval strategies
over a corpus of GCP cloud-native architecture documentation.
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Embedding Model", config.embedding_model_name)
col2.metric("Vector Dimensions", config.embedding_dim)
col3.metric("Index Type", "FAISS IndexFlatIP")
col4.metric("Corpus Size", "100 chunks")

st.divider()

st.subheader("Pipeline Architecture")

st.markdown("""
```
                        ┌─────────────────────────────────────────────────────┐
                        │              Context-Aware Retrieval Engine          │
                        └─────────────────────────────────────────────────────┘

  Raw Text Corpus                                              User Query
       │                                                           │
       ▼                                                           │
 ┌───────────┐    embed()    ┌──────────────────┐                 │
 │  Dataset  │──────────────▶│ SentenceTransformer│               │
 │  (JSON)   │               │  Embedder         │                │
 └───────────┘               │ (MockVertex API)  │                │
                             └────────┬──────────┘                │
                                      │ L2-norm vectors            │
                                      ▼                           │
                             ┌──────────────────┐                 │
                             │  FAISS IndexFlatIP│                 │
                             │  (Cosine Sim)     │                 │
                             └────────┬──────────┘                 │
                                      │                            │
                    ┌─────────────────┴──────────────────┐        │
                    │                                    │         │
              Strategy A                          Strategy B       │
           (Raw Vector Search)              (AI-Enhanced Retrieval)│
                    │                                    │         │
                    │                          ┌─────────┴──────┐  │
                    │                          │ MockVertex     │◄─┘
                    │                          │ GenerativeModel│
                    │                          │ (query expand) │
                    │                          └────────┬───────┘
                    │                                   │ expanded query
                    │                          embed()  │
                    │                                   ▼
                    │                          ┌────────────────┐
                    │                          │  FAISS search  │
                    │                          └────────┬───────┘
                    │                                   │
                    └──────────────┬────────────────────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │ RetrievalResult│
                          │ text + score   │
                          │ rank + chunk_id│
                          └────────────────┘
```
""")

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<span class="strategy-tag tag-a">Strategy A</span>', unsafe_allow_html=True)
    st.markdown("**Raw Vector Search**")
    st.markdown("""
    1. Embed the raw user query
    2. L2-normalise query vector
    3. FAISS `IndexFlatIP` inner product search
    4. Return top-k chunks by cosine similarity
    """)

with col_b:
    st.markdown('<span class="strategy-tag tag-b">Strategy B</span>', unsafe_allow_html=True)
    st.markdown("**AI-Enhanced Retrieval**")
    st.markdown("""
    1. Pass query to `MockVertexGenerativeModel.generate_content()`
    2. Expand with domain-specific technical synonyms
    3. Embed the enriched query
    4. FAISS search — bridges business language → technical documents
    """)

st.divider()
st.markdown("**Navigate using the sidebar** → Interactive Query | Benchmark Dashboard | Architecture")
