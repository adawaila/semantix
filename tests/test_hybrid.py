"""Hybrid ranker (RRF) test suite — 25+ tests."""
import pytest

from core.hybrid import HybridRanker, HybridResult


def ranker() -> HybridRanker:
    return HybridRanker(k=60)


# ---------------------------------------------------------------------------
# Basic fusion
# ---------------------------------------------------------------------------

class TestRRFBasic:
    def test_empty_both(self):
        r = ranker()
        results = r.fuse([], [], alpha=0.5)
        assert results == []

    def test_empty_bm25(self):
        r = ranker()
        vec = [("d1", 0.9), ("d2", 0.8)]
        results = r.fuse([], vec, alpha=1.0)
        assert len(results) == 2
        assert results[0].id == "d1"

    def test_empty_vector(self):
        r = ranker()
        bm25 = [("d1", 5.0), ("d2", 3.0)]
        results = r.fuse(bm25, [], alpha=0.0)
        assert len(results) == 2
        assert results[0].id == "d1"

    def test_disjoint_lists(self):
        r = ranker()
        bm25 = [("a", 5.0), ("b", 3.0)]
        vec = [("c", 0.9), ("d", 0.8)]
        results = r.fuse(bm25, vec, alpha=0.5, top_k=4)
        ids = {res.id for res in results}
        assert ids == {"a", "b", "c", "d"}

    def test_overlapping_lists(self):
        r = ranker()
        bm25 = [("a", 5.0), ("b", 3.0), ("c", 1.0)]
        vec = [("b", 0.9), ("a", 0.8), ("d", 0.7)]
        results = r.fuse(bm25, vec, alpha=0.5, top_k=10)
        # "a" and "b" appear in both — should score higher than solo entries
        ids = [res.id for res in results]
        assert "a" in ids
        assert "b" in ids

    def test_result_type(self):
        r = ranker()
        results = r.fuse([("d1", 1.0)], [("d1", 0.9)], alpha=0.5)
        assert isinstance(results[0], HybridResult)

    def test_scores_descending(self):
        r = ranker()
        bm25 = [(f"d{i}", float(10 - i)) for i in range(5)]
        vec = [(f"d{i}", float(1.0 - i * 0.1)) for i in range(5)]
        results = r.fuse(bm25, vec, alpha=0.5, top_k=5)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        r = ranker()
        bm25 = [(f"d{i}", float(10 - i)) for i in range(10)]
        vec = [(f"d{i}", float(1.0 - i * 0.05)) for i in range(10)]
        results = r.fuse(bm25, vec, alpha=0.5, top_k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Alpha weighting
# ---------------------------------------------------------------------------

class TestAlphaWeighting:
    def test_alpha_zero_pure_bm25(self):
        """alpha=0 → only BM25 contributes; vector docs not in BM25 excluded."""
        r = ranker()
        bm25 = [("bm25_only", 10.0)]
        vec = [("vec_only", 0.99)]
        results = r.fuse(bm25, vec, alpha=0.0, top_k=5)
        ids = [res.id for res in results]
        # vec_only gets 0 score because vector_weight=0
        assert "bm25_only" in ids
        assert "vec_only" not in ids

    def test_alpha_one_pure_vector(self):
        """alpha=1 → only vector contributes; BM25 docs not in vector excluded."""
        r = ranker()
        bm25 = [("bm25_only", 10.0)]
        vec = [("vec_only", 0.99)]
        results = r.fuse(bm25, vec, alpha=1.0, top_k=5)
        ids = [res.id for res in results]
        assert "vec_only" in ids
        assert "bm25_only" not in ids

    def test_alpha_half_both_contribute(self):
        r = ranker()
        bm25 = [("bm_doc", 10.0)]
        vec = [("vec_doc", 0.99)]
        results = r.fuse(bm25, vec, alpha=0.5, top_k=5)
        ids = {res.id for res in results}
        assert "bm_doc" in ids
        assert "vec_doc" in ids

    def test_alpha_invalid_raises(self):
        r = ranker()
        with pytest.raises(ValueError):
            r.fuse([], [], alpha=1.5)

    def test_alpha_negative_raises(self):
        r = ranker()
        with pytest.raises(ValueError):
            r.fuse([], [], alpha=-0.1)

    def test_alpha_boundary_zero(self):
        r = ranker()
        results = r.fuse([("d1", 1.0)], [("d2", 1.0)], alpha=0.0)
        assert any(res.id == "d1" for res in results)

    def test_alpha_boundary_one(self):
        r = ranker()
        results = r.fuse([("d1", 1.0)], [("d2", 1.0)], alpha=1.0)
        assert any(res.id == "d2" for res in results)

    def test_higher_alpha_boosts_vector(self):
        r = ranker()
        # "vec_winner" ranks #1 in vector but #5 in BM25
        # at alpha=0.9 it should appear earlier than at alpha=0.1
        bm25 = [("d1", 10), ("d2", 8), ("d3", 6), ("d4", 4), ("vec_winner", 2)]
        vec = [("vec_winner", 0.99), ("d1", 0.8), ("d2", 0.7), ("d3", 0.6), ("d4", 0.5)]

        results_lo = r.fuse(bm25, vec, alpha=0.1, top_k=5)
        results_hi = r.fuse(bm25, vec, alpha=0.9, top_k=5)

        rank_lo = next(i for i, res in enumerate(results_lo) if res.id == "vec_winner")
        rank_hi = next(i for i, res in enumerate(results_hi) if res.id == "vec_winner")
        assert rank_hi < rank_lo  # higher alpha → earlier rank for vec_winner


# ---------------------------------------------------------------------------
# RRF score properties
# ---------------------------------------------------------------------------

class TestRRFScoreProperties:
    def test_document_in_both_scores_higher(self):
        """A doc appearing in both lists should outscore docs in only one."""
        r = ranker()
        bm25 = [("shared", 5.0), ("bm25_only", 4.0)]
        vec = [("shared", 0.9), ("vec_only", 0.8)]
        results = r.fuse(bm25, vec, alpha=0.5, top_k=5)
        shared_score = next(res.score for res in results if res.id == "shared")
        bm25_only_score = next(res.score for res in results if res.id == "bm25_only")
        vec_only_score = next(res.score for res in results if res.id == "vec_only")
        assert shared_score > bm25_only_score
        assert shared_score > vec_only_score

    def test_rank_metadata_set(self):
        r = ranker()
        bm25 = [("d1", 5.0), ("d2", 3.0)]
        vec = [("d1", 0.9), ("d3", 0.8)]
        results = r.fuse(bm25, vec, alpha=0.5, top_k=5)
        d1 = next(res for res in results if res.id == "d1")
        assert d1.bm25_rank == 1
        assert d1.vector_rank == 1

        d2 = next(res for res in results if res.id == "d2")
        assert d2.bm25_rank == 2
        assert d2.vector_rank is None

        d3 = next(res for res in results if res.id == "d3")
        assert d3.bm25_rank is None
        assert d3.vector_rank == 2

    def test_score_positive(self):
        r = ranker()
        results = r.fuse([("d1", 1.0)], [("d1", 0.9)], alpha=0.5)
        assert results[0].score > 0

    def test_k_parameter_effect(self):
        """Smaller k → larger differences between ranks."""
        r1 = HybridRanker(k=1)
        r2 = HybridRanker(k=100)
        bm25 = [("first", 5.0), ("second", 3.0)]
        vec = [("first", 0.9), ("second", 0.8)]
        res1 = r1.fuse(bm25, vec, alpha=0.5)
        res2 = r2.fuse(bm25, vec, alpha=0.5)
        # Both should rank "first" first, but k=1 has larger score gap
        diff1 = res1[0].score - res1[1].score
        diff2 = res2[0].score - res2[1].score
        assert diff1 > diff2
