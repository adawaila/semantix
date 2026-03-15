"""Concurrent read/write tests under load."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.bm25 import BM25Index
from core.collection import Collection, CollectionConfig, IndexType
from core.vector import BruteForceIndex, Vector

DIM = 16


def rand_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_config(name: str = "concurrent") -> CollectionConfig:
    return CollectionConfig(
        name=name,
        embedding_field="text",
        embedding_dim=DIM,
        index_type=IndexType.BRUTE_FORCE,
    )


# ---------------------------------------------------------------------------
# BruteForce concurrent tests
# ---------------------------------------------------------------------------

class TestBruteForceConurrent:
    def test_concurrent_adds_no_crash(self):
        idx = BruteForceIndex(dim=DIM)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 50):
                    idx.add(Vector(id=str(i), data=rand_emb(i)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_concurrent_reads_during_writes(self):
        idx = BruteForceIndex(dim=DIM)
        for i in range(20):
            idx.add(Vector(id=str(i), data=rand_emb(i)))

        errors = []
        results = []

        def reader():
            try:
                q = rand_emb(99)
                r = idx.search(q, top_k=5)
                results.append(len(r))
            except Exception as e:
                errors.append(e)

        def writer(i):
            try:
                idx.add(Vector(id=f"w{i}", data=rand_emb(i + 100)))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# BM25 concurrent tests
# ---------------------------------------------------------------------------

class TestBM25Concurrent:
    def test_concurrent_adds_and_searches(self):
        idx = BM25Index()
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 30):
                    idx.add(f"d{i}", f"document {i} content search test")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                idx.search("document content")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer, args=(i * 30,)))
        for _ in range(10):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# Collection concurrent tests
# ---------------------------------------------------------------------------

class TestCollectionConcurrent:
    def test_concurrent_ingest_and_search(self):
        col = Collection(make_config())
        # Pre-populate
        for i in range(20):
            col.add({"id": str(i), "text": f"document {i} content"})

        errors = []
        search_results = []

        def ingest_worker(start):
            try:
                for i in range(start, start + 20):
                    col.add({"id": f"new_{i}", "text": f"new document {i}"}, embedding=rand_emb(i))
            except Exception as e:
                errors.append(e)

        def search_worker():
            try:
                r = col.search("document content", alpha=0.0, top_k=5)
                search_results.append(len(r))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=ingest_worker, args=(i * 20,)))
        for _ in range(10):
            threads.append(threading.Thread(target=search_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Final doc count should be >= initial + newly ingested
        assert col.stats()["doc_count"] >= 20

    def test_concurrent_deletes_no_crash(self):
        col = Collection(make_config())
        ids = [str(i) for i in range(50)]
        for id_ in ids:
            col.add({"id": id_, "text": f"delete me {id_}"})

        errors = []

        def delete_worker(ids_subset):
            for id_ in ids_subset:
                try:
                    col.delete(id_)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=delete_worker, args=(ids[i::4],))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_high_read_concurrency(self):
        col = Collection(make_config())
        for i in range(30):
            col.add({"id": str(i), "text": f"search content {i}"})

        errors = []
        results = []

        def searcher():
            try:
                r = col.search("search content", alpha=0.0, top_k=5)
                results.append(len(r))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=searcher) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 20
        assert all(r > 0 for r in results)
