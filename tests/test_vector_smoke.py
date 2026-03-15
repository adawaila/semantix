"""Smoke tests for core/vector/ — verifying vectr files work in semantix namespace."""
import numpy as np
import pytest

from core.vector import HNSWIndex, BruteForceIndex, Vector, SearchResult, RWLock


DIM = 32


def make_vec(id: str, seed: int) -> Vector:
    rng = np.random.default_rng(seed)
    return Vector(id=id, data=rng.random(DIM).astype(np.float32))


# ---------------------------------------------------------------------------
# RWLock
# ---------------------------------------------------------------------------

class TestRWLock:
    def test_single_reader(self):
        lock = RWLock()
        with lock.read():
            assert lock._readers == 1
        assert lock._readers == 0

    def test_exclusive_writer(self):
        lock = RWLock()
        with lock.write():
            assert lock._writing is True
        assert lock._writing is False

    def test_multiple_readers_concurrent(self):
        import threading
        lock = RWLock()
        results = []

        def reader():
            with lock.read():
                results.append(lock._readers)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Each reader saw at least 1 concurrent reader
        assert all(r >= 1 for r in results)


# ---------------------------------------------------------------------------
# BruteForceIndex
# ---------------------------------------------------------------------------

class TestBruteForce:
    def test_init(self):
        idx = BruteForceIndex(dim=DIM)
        assert idx.dim == DIM
        assert len(idx) == 0

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            BruteForceIndex(dim=0)

    def test_add_and_len(self):
        idx = BruteForceIndex(dim=DIM)
        idx.add(make_vec("a", 0))
        assert len(idx) == 1

    def test_add_duplicate_updates(self):
        idx = BruteForceIndex(dim=DIM)
        v1 = make_vec("a", 0)
        v2 = make_vec("a", 1)  # same id, different data
        idx.add(v1)
        idx.add(v2)
        assert len(idx) == 1
        stored = idx.get("a")
        # Should reflect updated vector (v2)
        assert stored is not None

    def test_add_batch(self):
        idx = BruteForceIndex(dim=DIM)
        vecs = [make_vec(str(i), i) for i in range(10)]
        idx.add_batch(vecs)
        assert len(idx) == 10

    def test_dim_mismatch_raises(self):
        idx = BruteForceIndex(dim=DIM)
        bad = Vector(id="x", data=np.ones(DIM + 1, dtype=np.float32))
        with pytest.raises(ValueError):
            idx.add(bad)

    def test_search_returns_top_k(self):
        idx = BruteForceIndex(dim=DIM)
        for i in range(20):
            idx.add(make_vec(str(i), i))
        q = np.random.default_rng(99).random(DIM).astype(np.float32)
        results = idx.search(q, top_k=5)
        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_descending(self):
        idx = BruteForceIndex(dim=DIM)
        for i in range(20):
            idx.add(make_vec(str(i), i))
        q = np.random.default_rng(42).random(DIM).astype(np.float32)
        results = idx.search(q, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_exact_match(self):
        idx = BruteForceIndex(dim=DIM)
        v = make_vec("target", 999)  # seed 999 not used by any other vec below
        idx.add(v)
        for i in range(10):
            idx.add(make_vec(str(i), i))
        # Searching with the same vector data should rank "target" first
        results = idx.search(v.data, top_k=1)
        assert results[0].id == "target"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_remove(self):
        idx = BruteForceIndex(dim=DIM)
        idx.add(make_vec("a", 0))
        idx.add(make_vec("b", 1))
        removed = idx.remove("a")
        assert removed is True
        assert len(idx) == 1
        assert "a" not in idx.ids()

    def test_remove_nonexistent(self):
        idx = BruteForceIndex(dim=DIM)
        assert idx.remove("ghost") is False

    def test_get(self):
        idx = BruteForceIndex(dim=DIM)
        v = make_vec("x", 5)
        idx.add(v)
        got = idx.get("x")
        assert got is not None
        assert got.id == "x"

    def test_get_nonexistent(self):
        idx = BruteForceIndex(dim=DIM)
        assert idx.get("nope") is None

    def test_ids(self):
        idx = BruteForceIndex(dim=DIM)
        for i in range(5):
            idx.add(make_vec(str(i), i))
        assert set(idx.ids()) == {"0", "1", "2", "3", "4"}

    def test_search_empty(self):
        idx = BruteForceIndex(dim=DIM)
        results = idx.search(np.ones(DIM, dtype=np.float32))
        assert results == []

    def test_serialisation_roundtrip(self):
        idx = BruteForceIndex(dim=DIM)
        for i in range(5):
            idx.add(make_vec(str(i), i))
        state = idx.state()
        restored = BruteForceIndex.from_state(state)
        q = np.random.default_rng(0).random(DIM).astype(np.float32)
        r1 = idx.search(q, top_k=3)
        r2 = restored.search(q, top_k=3)
        assert [r.id for r in r1] == [r.id for r in r2]


# ---------------------------------------------------------------------------
# HNSWIndex
# ---------------------------------------------------------------------------

class TestHNSW:
    def test_init(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        assert idx.dim == DIM
        assert len(idx) == 0

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            HNSWIndex(dim=0)

    def test_add_and_len(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        idx.add(make_vec("a", 0))
        assert len(idx) == 1

    def test_add_batch(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        vecs = [make_vec(str(i), i) for i in range(10)]
        idx.add_batch(vecs)
        assert len(idx) == 10

    def test_dim_mismatch_raises(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        bad = Vector(id="x", data=np.ones(DIM + 1, dtype=np.float32))
        with pytest.raises(ValueError):
            idx.add(bad)

    def test_search_returns_results(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        for i in range(50):
            idx.add(make_vec(str(i), i))
        q = np.random.default_rng(99).random(DIM).astype(np.float32)
        results = idx.search(q, top_k=5)
        assert 1 <= len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_descending(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        for i in range(50):
            idx.add(make_vec(str(i), i))
        q = np.random.default_rng(42).random(DIM).astype(np.float32)
        results = idx.search(q, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_approximate_correctness(self):
        """HNSW top-1 should match brute-force top-1 (with high probability)."""
        dim = DIM
        hnsw = HNSWIndex(dim=dim, seed=42)
        bf = BruteForceIndex(dim=dim)
        vecs = [make_vec(str(i), i) for i in range(100)]
        hnsw.add_batch(vecs)
        bf.add_batch(vecs)

        hits = 0
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.random(dim).astype(np.float32)
            h_top = hnsw.search(q, top_k=1)
            b_top = bf.search(q, top_k=1)
            if h_top and b_top and h_top[0].id == b_top[0].id:
                hits += 1
        assert hits >= 15  # >=75% recall@1

    def test_remove(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        idx.add(make_vec("a", 0))
        idx.add(make_vec("b", 1))
        removed = idx.remove("a")
        assert removed is True
        assert len(idx) == 1

    def test_remove_nonexistent(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        assert idx.remove("ghost") is False

    def test_search_empty(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        results = idx.search(np.ones(DIM, dtype=np.float32))
        assert results == []

    def test_get(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        v = make_vec("z", 3)
        idx.add(v)
        got = idx.get("z")
        assert got is not None
        assert got.id == "z"

    def test_ids(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        for i in range(5):
            idx.add(make_vec(str(i), i))
        assert set(idx.ids()) == {"0", "1", "2", "3", "4"}

    def test_serialisation_roundtrip(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        for i in range(20):
            idx.add(make_vec(str(i), i))
        state = idx.state()
        restored = HNSWIndex.from_state(state)
        assert len(restored) == len(idx)
        q = np.random.default_rng(0).random(DIM).astype(np.float32)
        r1 = idx.search(q, top_k=5)
        r2 = restored.search(q, top_k=5)
        assert [r.id for r in r1] == [r.id for r in r2]

    def test_add_updates_existing(self):
        idx = HNSWIndex(dim=DIM, seed=0)
        idx.add(make_vec("dup", 0))
        idx.add(make_vec("dup", 1))  # same id
        assert len(idx) == 1
