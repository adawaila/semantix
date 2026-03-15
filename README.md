# semantix

**Self-hostable hybrid search engine — BM25 + vector search in one API.**

The vector index is powered by **[vectr](https://github.com/adawaila/vectr)**, a from-scratch HNSW implementation built in pure Python + NumPy. semantix and vectr are part of the same ecosystem: semantix is the full product built on top of vectr's indexing primitives. The HNSW index, BruteForce index, and RWLock from vectr are copied directly into `core/vector/` and used as-is — no reimplementation.

---

## Quick Start

```bash
docker compose up
```

```python
from sdk import Semantix

client = Semantix("http://localhost:8000")
client.create_collection("products", embedding_field="description")
client.ingest("products", [
    {"id": "1", "description": "wireless headphones with noise cancelling"},
])
results = client.search("products", "wireless headphones", top_k=10, alpha=0.5)
```

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
│  │(Next.js) │   │  + Job Track │   │  ┌────────────────┐  │    │
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
  └── core/vector/hnsw.py        HNSW multi-layer ANN graph
  └── core/vector/brute_force.py  Exact cosine similarity
  └── core/vector/rwlock.py       Writer-preferring RWLock
```

---

## Benchmark Results *(n=10,000, dim=384, CPU)*

| Mode    | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|----------|----------|----------|
| BM25    | 2.37     | 6.26     | 7.28     |
| Vector  | 2.26     | 3.42     | 4.47     |
| Hybrid  | 4.61     | 7.01     | 9.56     |

HNSW Recall@10 vs BruteForce: **98.4%**
Ingestion: **173 docs/sec** (embed) · **80 docs/sec** (HNSW index)

---

## vs. Algolia / Typesense

| Feature           | semantix  | Algolia   | Typesense |
|-------------------|-----------|-----------|-----------|
| Self-hostable     | ✅        | ❌        | ✅        |
| Hybrid search     | ✅ RRF    | ✅        | ✅        |
| Offline embedding | ✅        | ❌        | ❌        |
| Open source       | ✅ MIT    | ❌        | ✅ GPL-3  |
| No ext. index deps| ✅ NumPy  | ❌        | ❌        |
| Cost              | Free      | $$$       | Free/Paid |

---

See [`docs/README.md`](docs/README.md) for full documentation.

## License

MIT
