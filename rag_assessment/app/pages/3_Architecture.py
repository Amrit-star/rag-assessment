"""Architecture, similarity metric theory, and Vertex AI migration plan."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Architecture & Migration", page_icon="📐", layout="wide")

st.title("📐 Architecture & Migration")

tab_similarity, tab_migration, tab_design = st.tabs([
    "Similarity Metrics", "Vertex AI Migration", "Design Decisions"
])

# ── Tab 1: Similarity Metrics ─────────────────────────────────────────────────
with tab_similarity:
    st.subheader("Cosine Similarity vs Euclidean Distance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Euclidean Distance (L2)")
        st.latex(r"d(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}")
        st.markdown("""
        **Measures:** Geometric distance between two points in vector space.

        **Problem for text:** Euclidean distance is sensitive to vector **magnitude**.
        A long document and a short document discussing the *same topic* will have
        very different magnitudes — their L2 distance will be large even though they
        are semantically identical.

        **Example failure:** A 500-word article on load balancing vs a 50-word summary
        on load balancing → large L2 distance, but near-identical meaning.
        """)
        st.error("❌ Not suitable for text embedding retrieval without normalisation.")

    with col2:
        st.markdown("#### Cosine Similarity")
        st.latex(r"\cos(\theta) = \frac{A \cdot B}{\|A\| \cdot \|B\|}")
        st.markdown("""
        **Measures:** The **angle** between two vectors — directional similarity,
        independent of magnitude.

        **Why it works for text:** Two documents with the same semantic content point
        in the same direction in embedding space regardless of their length.
        The cosine score is always in **[−1, 1]**, with 1 meaning identical direction.
        """)
        st.success("✅ Standard metric for text embedding retrieval.")

    st.divider()
    st.subheader("How This Project Uses It: IndexFlatIP on L2-Normalised Vectors")
    st.markdown("""
    When vectors are L2-normalised (unit vectors, ‖v‖ = 1), the cosine similarity
    simplifies to the **inner product**:
    """)
    st.latex(r"\cos(\theta) = \frac{A \cdot B}{\|A\| \cdot \|B\|} = A \cdot B \quad \text{when } \|A\| = \|B\| = 1")
    st.markdown("""
    `SentenceTransformerEmbedder` sets `normalize_embeddings=True`, producing unit vectors.
    `FAISSVectorStore` uses `IndexFlatIP` (inner product index) — no additional normalisation
    overhead at query time. The result: **cosine similarity at the speed of dot product.**

    This is the same approach Vertex AI Vector Search uses internally with ScaNN.
    """)
    st.code("""
# FAISSVectorStore: why IndexFlatIP, not IndexFlatL2
index = faiss.IndexFlatIP(dim)   # inner product on unit vectors = cosine similarity
faiss.normalize_L2(embeddings)   # guarantee unit vectors at ingest time
faiss.normalize_L2(query_vec)    # and at search time
scores, ids = index.search(query_vec, k)
# scores[i] ∈ [0, 1] — cosine similarity
    """, language="python")

# ── Tab 2: Vertex AI Migration ────────────────────────────────────────────────
with tab_migration:
    st.subheader("Production Migration to Vertex AI Vector Search (Matching Engine)")

    steps = [
        {
            "step": "1",
            "title": "Storage — Cloud Storage / BigQuery",
            "icon": "🗄️",
            "gcp_service": "Cloud Storage + BigQuery",
            "detail": """
Move text chunks and metadata from local JSON to scalable storage.

- Store raw chunks as `JSONL` files in **Cloud Storage** (multi-region bucket for durability)
- Load metadata into **BigQuery** for analytics (chunk stats, retrieval logs, A/B experiment results)
- This decouples storage from compute — chunks can be updated without reindexing
            """,
            "code": """
# Upload chunks to GCS
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("your-rag-corpus")
blob = bucket.blob("chunks/technical_context.jsonl")
blob.upload_from_filename("data/technical_context.json")
            """,
        },
        {
            "step": "2",
            "title": "Indexing — Vertex AI Vector Search (ScaNN)",
            "icon": "🔍",
            "gcp_service": "Vertex AI Vector Search",
            "detail": """
Replace local `FAISSVectorStore` with **Vertex AI Vector Search** (formerly Matching Engine).

- Uses **ScaNN** (Scalable Nearest Neighbours) — handles **billions of vectors** with sub-10ms latency
- In-memory FAISS is limited to single-machine RAM; ScaNN is a distributed managed service
- Supports `DOT_PRODUCT_DISTANCE` on normalised vectors (equivalent to cosine similarity)
- Index updates are streamed — no full reindex required for new chunks
            """,
            "code": """
from google.cloud import aiplatform

aiplatform.init(project="your-project", location="us-central1")

# Create index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-corpus-index",
    contents_delta_uri="gs://your-rag-corpus/embeddings/",
    dimensions=768,  # gecko embedding dim
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)
            """,
        },
        {
            "step": "3",
            "title": "Deployment — Index Endpoint via VPC Peering",
            "icon": "🌐",
            "gcp_service": "Vertex AI Index Endpoint",
            "detail": """
Deploy the index to an **Index Endpoint** peered to your VPC for secure, low-latency queries.

- **VPC Peering** keeps retrieval traffic private — no public internet exposure
- Endpoint autoscales replica count based on QPS load
- `match_service_client.find_neighbors()` replaces `FAISSVectorStore.search()`
            """,
            "code": """
# Deploy and query the endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-endpoint",
    network="projects/123/global/networks/your-vpc",
)
endpoint.deploy_index(index=index, deployed_index_id="rag_index_v1")

# Query (replaces FAISSVectorStore.search)
response = endpoint.find_neighbors(
    deployed_index_id="rag_index_v1",
    queries=[query_embedding],
    num_neighbors=3,
)
            """,
        },
        {
            "step": "4",
            "title": "Application — Swap Mocks for Real Vertex AI Clients",
            "icon": "🔄",
            "gcp_service": "Vertex AI SDK (real)",
            "detail": """
The **dependency-injection design** of `ContextAwareRetrievalEngine` means zero orchestration
changes are required. Only the concrete implementations are swapped:

- `MockVertexTextEmbeddingModel` → real `vertexai.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")`
- `MockVertexGenerativeModel` → real `vertexai.generativeai.GenerativeModel("gemini-1.5-flash")`
- `FAISSVectorStore` → `VertexAIVectorStore` adapter wrapping the Index Endpoint client

The orchestrator's `strategy_a_search` and `strategy_b_search` methods are untouched.
            """,
            "code": """
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generativeai import GenerativeModel

vertexai.init(project="your-project", location="us-central1")

# Drop-in replacements — same API surface as the mocks
real_embedder = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
real_expander = GenerativeModel("gemini-1.5-flash")

engine = ContextAwareRetrievalEngine(
    embedding_model=VertexEmbedderAdapter(real_embedder),  # thin adapter
    vector_store=VertexVectorStoreAdapter(endpoint),
    query_expander=VertexExpanderAdapter(real_expander),
)
            """,
        },
    ]

    for s in steps:
        with st.expander(f"{s['icon']} Step {s['step']}: {s['title']}  —  `{s['gcp_service']}`", expanded=True):
            col_detail, col_code = st.columns([1, 1])
            with col_detail:
                st.markdown(s["detail"])
            with col_code:
                st.code(s["code"], language="python")

# ── Tab 3: Design Decisions ───────────────────────────────────────────────────
with tab_design:
    st.subheader("Key Engineering Decisions")

    decisions = [
        ("Abstract Base Classes (ABCs)", """
`BaseEmbeddingModel`, `BaseVectorStore`, and `BaseQueryExpander` define the contracts.
`ContextAwareRetrievalEngine` depends only on these interfaces — never on concrete classes.
This is the **Dependency Inversion Principle**: high-level policy does not depend on
low-level detail. Swapping FAISS for Vertex AI requires changing one constructor argument,
not modifying the orchestrator.
        """),
        ("IndexFlatIP over IndexFlatL2", """
With L2-normalised embeddings, inner product is mathematically identical to cosine similarity
but computed with a single dot-product rather than a full Euclidean distance calculation.
Using `IndexFlatL2` on non-normalised vectors would give incorrect semantic rankings for
text retrieval. Using `IndexFlatL2` on normalised vectors would give correct rankings but
at unnecessary computational cost.
        """),
        ("Mock API Shape Fidelity", """
`MockVertexTextEmbeddingModel` returns `TextEmbedding` objects with `.values` and `.statistics`
attributes — the exact contract of the real Vertex SDK. Tests assert on the API contract,
not on implementation details. If the real SDK is wired in, no test changes are needed.
        """),
        ("Query Expansion via Keyword Rules", """
Rather than a lookup dictionary (brittle for unseen queries), the expander parses the query
for domain keywords and injects GCP-specific synonyms. This makes Strategy B demonstrably
better: business-language queries ("How do you keep users logged in?") are enriched with
technical terms ("JWT", "Memorystore", "stateless") that appear verbatim in the corpus,
closing the vocabulary gap that defeats raw vector search.
        """),
        ("RetrievalResult Dataclass", """
Results are typed `RetrievalResult` objects (text, score, rank, chunk_id) rather than tuples.
The `score` field carries the actual cosine similarity, making benchmark comparisons quantitative
rather than qualitative. Tests assert `score ∈ [0, 1]` and `results sorted descending`.
        """),
    ]

    for title, detail in decisions:
        with st.expander(f"**{title}**"):
            st.markdown(detail.strip())
