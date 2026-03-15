"""semantix benchmark suite.

Usage:
    python benchmark/benchmark.py --n 10000 --dim 384 --n-queries 100 --alpha 0.5

Reports:
- Ingestion throughput (docs/sec)
- Search latency p50/p95/p99 for BM25, vector, and hybrid
- Recall@10 of HNSW vs BruteForce baseline
- Saves results to benchmark/results/latest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Make sure we can import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.dataset import generate_dataset, generate_queries
from core.collection import Collection, CollectionConfig, IndexType
from embeddings import LocalEmbeddings


def percentile(arr: list[float], p: float) -> float:
    a = sorted(arr)
    idx = int(len(a) * p / 100)
    return a[min(idx, len(a) - 1)]


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant[:k])) / min(k, len(relevant))


def run_benchmark(
    n: int = 1000,
    dim: int = 384,
    n_queries: int = 50,
    alpha: float = 0.5,
    embed_batch_size: int = 32,
) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  semantix benchmark")
    print(f"  n={n}, dim={dim}, queries={n_queries}, alpha={alpha}")
    print(f"{'='*60}\n")

    # Use local embeddings
    print("Loading embedding model...")
    provider = LocalEmbeddings()
    actual_dim = provider.dim
    if actual_dim != dim:
        print(f"  Note: model dim={actual_dim} (overriding --dim={dim})")
        dim = actual_dim

    # Generate dataset
    print(f"Generating {n} synthetic products...")
    docs = generate_dataset(n=n)
    queries = generate_queries(n=n_queries)

    # ----------------------------------------------------------------
    # Setup collections
    # ----------------------------------------------------------------
    hnsw_col = Collection(CollectionConfig(
        name="bench_hnsw", embedding_field="description", embedding_dim=dim,
        index_type=IndexType.HNSW, hnsw_ef_search=50,
    ))
    bf_col = Collection(CollectionConfig(
        name="bench_bf", embedding_field="description", embedding_dim=dim,
        index_type=IndexType.BRUTE_FORCE,
    ))

    # ----------------------------------------------------------------
    # Ingestion benchmark
    # ----------------------------------------------------------------
    print(f"\n[1/4] Ingestion benchmark ({n} docs)...")
    texts = [str(doc.get("description", "")) for doc in docs]

    t0 = time.perf_counter()
    all_embs = provider.embed_batch(texts)
    embed_time = time.perf_counter() - t0
    embed_throughput = n / embed_time

    t0 = time.perf_counter()
    for i, doc in enumerate(docs):
        hnsw_col.add(doc, embedding=all_embs[i])
    hnsw_index_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i, doc in enumerate(docs):
        bf_col.add(doc, embedding=all_embs[i])
    bf_index_time = time.perf_counter() - t0

    total_ingest_time = embed_time + hnsw_index_time
    ingest_throughput = n / total_ingest_time

    print(f"  Embedding: {embed_throughput:.0f} docs/sec")
    print(f"  HNSW indexing: {n/hnsw_index_time:.0f} docs/sec")
    print(f"  BruteForce indexing: {n/bf_index_time:.0f} docs/sec")
    print(f"  Overall ingestion: {ingest_throughput:.0f} docs/sec")

    # ----------------------------------------------------------------
    # Embed queries
    # ----------------------------------------------------------------
    print(f"\n[2/4] Embedding {n_queries} queries...")
    q_embs = provider.embed_batch(queries)

    # ----------------------------------------------------------------
    # BM25 search benchmark
    # ----------------------------------------------------------------
    print(f"\n[3/4] Search benchmarks...")
    bm25_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        hnsw_col.search(q, query_embedding=None, top_k=10, alpha=0.0)
        bm25_latencies.append((time.perf_counter() - t0) * 1000)

    vec_latencies = []
    for q_emb in q_embs:
        t0 = time.perf_counter()
        hnsw_col.search("", query_embedding=q_emb, top_k=10, alpha=1.0)
        vec_latencies.append((time.perf_counter() - t0) * 1000)

    hybrid_latencies = []
    for q, q_emb in zip(queries, q_embs):
        t0 = time.perf_counter()
        hnsw_col.search(q, query_embedding=q_emb, top_k=10, alpha=alpha)
        hybrid_latencies.append((time.perf_counter() - t0) * 1000)

    print(f"  BM25   p50={percentile(bm25_latencies,50):.2f}ms  p95={percentile(bm25_latencies,95):.2f}ms  p99={percentile(bm25_latencies,99):.2f}ms")
    print(f"  Vector p50={percentile(vec_latencies,50):.2f}ms  p95={percentile(vec_latencies,95):.2f}ms  p99={percentile(vec_latencies,99):.2f}ms")
    print(f"  Hybrid p50={percentile(hybrid_latencies,50):.2f}ms  p95={percentile(hybrid_latencies,95):.2f}ms  p99={percentile(hybrid_latencies,99):.2f}ms")

    # ----------------------------------------------------------------
    # Recall@10: HNSW vs BruteForce
    # ----------------------------------------------------------------
    print(f"\n[4/4] Recall@10 (HNSW vs BruteForce)...")
    recalls = []
    for q_emb in q_embs:
        hnsw_ids = [r.id for r in hnsw_col._vec_index.search(q_emb, top_k=10)]
        bf_ids = [r.id for r in bf_col._vec_index.search(q_emb, top_k=10)]
        recalls.append(recall_at_k(hnsw_ids, bf_ids, k=10))

    mean_recall = float(np.mean(recalls))
    print(f"  Recall@10: {mean_recall:.4f} ({mean_recall*100:.1f}%)")

    # ----------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------
    results = {
        "config": {"n": n, "dim": dim, "n_queries": n_queries, "alpha": alpha},
        "ingestion": {
            "embed_throughput_docs_per_sec": round(embed_throughput, 1),
            "hnsw_index_throughput_docs_per_sec": round(n / hnsw_index_time, 1),
            "bf_index_throughput_docs_per_sec": round(n / bf_index_time, 1),
            "total_throughput_docs_per_sec": round(ingest_throughput, 1),
        },
        "search_latency_ms": {
            "bm25": {
                "p50": round(percentile(bm25_latencies, 50), 2),
                "p95": round(percentile(bm25_latencies, 95), 2),
                "p99": round(percentile(bm25_latencies, 99), 2),
            },
            "vector": {
                "p50": round(percentile(vec_latencies, 50), 2),
                "p95": round(percentile(vec_latencies, 95), 2),
                "p99": round(percentile(vec_latencies, 99), 2),
            },
            "hybrid": {
                "p50": round(percentile(hybrid_latencies, 50), 2),
                "p95": round(percentile(hybrid_latencies, 95), 2),
                "p99": round(percentile(hybrid_latencies, 99), 2),
            },
        },
        "recall": {
            "hnsw_recall_at_10": round(mean_recall, 4),
        },
    }

    print(f"\n{'='*60}")
    print("  Benchmark complete.")
    print(f"{'='*60}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="semantix benchmark")
    parser.add_argument("--n", type=int, default=1000, help="Number of documents")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension (informational)")
    parser.add_argument("--n-queries", type=int, default=50, help="Number of test queries")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha weight")
    args = parser.parse_args()

    results = run_benchmark(
        n=args.n,
        dim=args.dim,
        n_queries=args.n_queries,
        alpha=args.alpha,
    )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "latest.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
