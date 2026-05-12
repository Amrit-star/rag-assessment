# Context-Aware Retrieval Engine

**Senior Gen AI Assessment — Semantic RAG & Vector Search**

A production-grade RAG pipeline comparing two retrieval strategies over a corpus of GCP cloud-native architecture documentation. Built with dependency-injected components, FAISS vector search, and a mocked Vertex AI SDK layer.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests
pytest tests/ -v

# 3. Generate benchmark report (writes retrieval_benchmark.md + benchmark_results.json)
python run_benchmark.py

# 4. Launch Streamlit demo
PYTHONPATH=. streamlit run app/Home.py
# On Windows PowerShell:
# $env:PYTHONPATH="."; streamlit run app/Home.py
```

No API keys required. No GCP credentials needed. Everything runs locally.

---

## Project Structure

```
rag-assessment/
├── src/
│   ├── interfaces.py       # ABCs: BaseEmbeddingModel, BaseVectorStore, BaseQueryExpander
│   ├── config.py           # Centralised config (model name, dim, paths)
│   ├── embeddings.py       # SentenceTransformerEmbedder + MockVertexTextEmbeddingModel
│   ├── vector_store.py     # FAISSVectorStore (IndexFlatIP, save/load)
│   ├── query_expansion.py  # MockVertexGenerativeModel (domain-aware expansion)
│   └── orchestrator.py     # ContextAwareRetrievalEngine (dependency-injected)
├── tests/
│   ├── conftest.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_query_expansion.py
│   └── test_orchestrator.py
├── data/
│   └── technical_context.json   # 100 GCP architecture chunks
├── app/
│   ├── Home.py                  # Streamlit landing page
│   ├── engine_factory.py        # Cached engine initialisation
│   └── pages/
│       ├── 1_Interactive_Query.py
│       ├── 2_Benchmark_Dashboard.py
│       └── 3_Architecture.py
├── run_benchmark.py             # CLI benchmark runner
├── retrieval_benchmark.md       # Generated benchmark output (committed)
└── benchmark_results.json       # Machine-readable results (committed)
```

---

## Architecture

```
Raw Text Corpus → SentenceTransformerEmbedder → FAISS IndexFlatIP
                                                        │
                              ┌─────────────────────────┴──────────────────────┐
                              │                                                 │
                        Strategy A                                        Strategy B
                    (Raw Vector Search)                          (AI-Enhanced Retrieval)
                              │                                                 │
                    embed(raw_query)                    MockVertexGenerativeModel
                              │                                expand(query)
                              │                                     │
                              │                          embed(expanded_query)
                              │                                     │
                              └──────────────┬──────────────────────┘
                                             │
                                   list[RetrievalResult]
                                   (text, score, rank, chunk_id)
```

---

## Similarity Metric: Cosine vs Euclidean

### Why Cosine Similarity

Euclidean distance (L2) measures the geometric distance between two points in vector space. For text embeddings this is problematic: a 500-word article and a 50-word summary on the same topic will have very different vector magnitudes, resulting in a large L2 distance despite near-identical semantics.

**Cosine similarity** measures the *angle* between vectors — directional similarity, independent of magnitude:

```
cos(θ) = (A · B) / (‖A‖ · ‖B‖)
```

For unit vectors (‖A‖ = ‖B‖ = 1), this reduces to the inner product:

```
cos(θ) = A · B
```

### Implementation Choice: `IndexFlatIP` on L2-Normalised Vectors

`SentenceTransformerEmbedder` produces unit vectors (`normalize_embeddings=True`). `FAISSVectorStore` uses `IndexFlatIP` (inner product index). On unit vectors, inner product equals cosine similarity — giving semantically correct rankings at dot-product speed.

Using `IndexFlatL2` on non-normalised vectors would produce incorrect rankings. Using `IndexFlatL2` on normalised vectors would produce correct rankings but at unnecessary cost. `IndexFlatIP` on normalised vectors is the correct and efficient choice.

> This is the same approach Vertex AI Vector Search (ScaNN) uses internally.

---

## Production Migration: Vertex AI Vector Search

### Step 1 — Storage: Cloud Storage + BigQuery

Move text chunks to Cloud Storage (JSONL, multi-region) and metadata to BigQuery for analytics and retrieval logging.

### Step 2 — Indexing: Vertex AI Vector Search (ScaNN)

Replace `FAISSVectorStore` with a Vertex AI Vector Search index. ScaNN handles billions of vectors with sub-10ms latency. In-memory FAISS is bounded by single-machine RAM. Index updates are streamed, eliminating full reindexing for new chunks.

```python
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-corpus-index",
    contents_delta_uri="gs://your-corpus/embeddings/",
    dimensions=768,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)
```

### Step 3 — Deployment: Index Endpoint via VPC Peering

Deploy to a Vertex AI Index Endpoint peered to your VPC. Traffic stays private, the endpoint autoscales on QPS, and `find_neighbors()` replaces `FAISSVectorStore.search()`.

### Step 4 — Application: Swap Mocks for Real Clients

The `ContextAwareRetrievalEngine` depends only on the `Base*` ABCs. Swapping implementations requires changing constructor arguments only — no orchestration logic changes:

| Mock (local) | Production |
|---|---|
| `MockVertexTextEmbeddingModel` | `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` |
| `MockVertexGenerativeModel` | `GenerativeModel("gemini-1.5-flash")` |
| `FAISSVectorStore` | `VertexAIVectorStoreAdapter(endpoint)` |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Key assertions:
- Embedding output shape is `(n, 384)` with unit norm
- `MockVertexTextEmbeddingModel` returns objects with `.values` and `.statistics` (Vertex API contract)
- `FAISSVectorStore` scores are in `[0, 1]`, sorted descending, exact match is rank 1
- `strategy_a_search` calls `generate_content` **zero times**
- `strategy_b_search` calls `generate_content` **exactly once**
