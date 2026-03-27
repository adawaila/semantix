# semantix

**Self-hostable hybrid search engine — BM25 + vector search in one API.**

The vector index is powered by **[vectr](https://github.com/adawaila/vectr)**, a from-scratch HNSW implementation built in pure Python + NumPy. semantix and vectr are part of the same ecosystem: semantix is the full product built on top of vectr's indexing primitives. The HNSW index, BruteForce index, and RWLock from vectr are copied directly into `core/vector/` and used as-is — no reimplementation.

---

## Quick Start

```bash
# Run everything with a single command
docker compose up
```

Then use the Python SDK:

```python
from sdk import Semantix

client = Semantix("http://localhost:8000")
client.create_collection("products", embedding_field="description")
client.ingest("products", [
    {"id": "1", "description": "wireless headphones with noise cancelling"},
    {"id": "2", "description": "bluetooth speaker, 360° sound"},
])
results = client.search("products", "wireless headphones", top_k=10, alpha=0.5)
```

Dashboard: open `http://localhost:3000`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          semantix                               │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │  Client  │──▶│  FastAPI API │──▶│  Collection Engine   │    │
│  │  (SDK)   │   │  (8000)      │   │                      │    │
│  └──────────┘   └──────────────┘   │  ┌────────────────┐  │    │
│                        │           │  │  BM25 Index    │  │    │
│  ┌──────────┐   ┌──────────────┐   │  │  (inverted)    │  │    │
│  │Dashboard │   │  Redis Queue │   │  └────────────────┘  │    │
│  │(Next.js) │   │  + Job Tract │   │  ┌────────────────┐  │    │
│  └──────────┘   └──────────────┘   │  │  HNSW Index    │  │    │
│                        │           │  │  (from vectr)  │  │    │
│                 ┌──────────────┐   │  └────────────────┘  │    │
│                 │  Workers (4) │──▶│  ┌────────────────┐  │    │
│                 │  Chunk→Embed │   │  │  RRF Ranker    │  │    │
│                 │  →Index      │   │  │  (hybrid)      │  │    │
│                 └──────────────┘   │  └────────────────┘  │    │
│                                    └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

Vector backend: github.com/adawaila/vectr
  └── core/vector/hnsw.py       (HNSW multi-layer ANN graph)
  └── core/vector/brute_force.py (exact cosine similarity)
  └── core/vector/rwlock.py      (writer-preferring RWLock)
```

---

## Benchmark Results

Measured on CPU (Intel Core, n=10,000 docs, all-MiniLM-L6-v2, dim=384).

### Ingestion Throughput

| Stage                | Throughput       |
|----------------------|-----------------|
| Embedding (batch)    | 173 docs/sec    |
| HNSW indexing        | 80 docs/sec     |
| BruteForce indexing  | 195 docs/sec    |
| End-to-end (embed+HNSW) | 55 docs/sec |

### Search Latency (ms) — 100 queries

| Mode       | p50   | p95   | p99   |
|------------|-------|-------|-------|
| BM25       | 2.37  | 6.26  | 7.28  |
| Vector     | 2.26  | 3.42  | 4.47  |
| Hybrid     | 4.61  | 7.01  | 9.56  |

### Recall@10

| Index      | Recall@10 |
|------------|-----------|
| HNSW       | 98.4%     |
| BruteForce | 100%      |

---

## API Reference

```
Collections:
  POST   /collections                  Create a collection
  GET    /collections                  List all collections
  GET    /collections/{name}           Get collection stats
  DELETE /collections/{name}           Delete collection

Documents:
  POST   /collections/{name}/documents         Sync ingest (immediate)
  POST   /collections/{name}/documents/bulk    Async bulk ingest (returns job_id)
  DELETE /collections/{name}/documents/{id}    Delete document

Search:
  POST   /collections/{name}/search
    body: { query, top_k, alpha, filters }

Jobs:
  GET    /jobs/{job_id}                Get job status
  WS     /jobs/{job_id}/stream         Live progress stream
```

### Search body

```json
{
  "query": "wireless headphones",
  "top_k": 10,
  "alpha": 0.5,
  "filters": {
    "category": "electronics",
    "price": { "$lte": 100 }
  }
}
```

**`alpha`**: `0.0` = pure BM25 · `1.0` = pure vector · `0.5` = balanced hybrid

---

## Python SDK

```python
from sdk import Semantix, AsyncSemantix

# Sync
with Semantix("http://localhost:8000") as client:
    client.create_collection("articles", embedding_field="body")
    client.ingest("articles", docs)
    results = client.search("articles", "machine learning", top_k=5, alpha=0.7)
    for r in results.results:
        print(r.id, r.score, r.document)

# Async
async with AsyncSemantix("http://localhost:8000") as client:
    await client.ingest("articles", docs)
    results = await client.search("articles", "neural networks")
```

---

## Configuration

| Variable          | Default              | Description                      |
|-------------------|----------------------|----------------------------------|
| `EMBED_PROVIDER`  | `local`              | `local` or `openai`              |
| `OPENAI_API_KEY`  | —                    | Required if provider=openai      |
| `DATA_DIR`        | `./data`             | Persistence directory            |
| `REDIS_URL`       | `redis://localhost:6379` | Redis for job queue          |
| `WORKERS`         | `4`                  | Ingestion worker processes       |
| `API_PORT`        | `8000`               | FastAPI port                     |
| `DASHBOARD_PORT`  | `3000`               | Next.js dashboard port           |

---

## Running from Source

```bash
# Install
pip install -e ".[dev]"

# Start API
uvicorn api.app:app --reload

# Start workers
python -m ingestion.worker --redis redis://localhost:6379 --workers 4

# Run tests
pytest tests/ -q

# Benchmark
python benchmark/benchmark.py --n 10000 --n-queries 100
```

---

## Architecture Notes

### Why vectr?

semantix uses the HNSW and BruteForce indices from [vectr](https://github.com/adawaila/vectr) directly. This is intentional — vectr provides:
- **Multi-layer HNSW** with diversity-heuristic neighbour selection
- **Writer-preferring RWLock** preventing reader starvation
- **Pure NumPy hot paths** — no FAISS, no Annoy, no external ANN libs

semantix adds BM25, hybrid fusion, an ingestion pipeline, REST API, SDK, and dashboard on top.

### Hybrid Search via RRF

Reciprocal Rank Fusion (Cormack et al., 2009):

```
score(d) = alpha × 1/(k + rank_vector(d))
         + (1-alpha) × 1/(k + rank_bm25(d))
```

k=60 (empirically optimal), configurable per-query alpha.

### Persistence

Atomic pickle: write to `.tmp` → `os.replace()`. Collections survive crashes cleanly.

---

## Contributing

1. Fork the repo
2. Run tests: `pytest tests/ -q` (227 tests, must all pass)
3. Add tests for any new feature
4. Open a PR with a clear description

Please keep the link to vectr visible — semantix is built *on top of* vectr, not as a replacement.

