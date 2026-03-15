"""Hybrid ranker using Reciprocal Rank Fusion (RRF).

RRF merges two ranked lists (BM25 and vector) into a single ranking:
    RRF_score(d) = sum_r( 1 / (k + rank_r(d)) )

where k=60 (Cormack et al., 2009 SIGIR) and rank_r(d) is the 1-based rank
of document d in ranker r (0 if absent).

Alpha parameter linearly blends the two result sets before fusion:
    alpha = 0.0  →  pure BM25  (vector results ignored)
    alpha = 1.0  →  pure vector (BM25 results ignored)
    alpha = 0.5  →  equal weight hybrid
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HybridResult:
    """A single result from hybrid search."""
    id: str
    score: float
    bm25_rank: int | None = None    # 1-based rank in BM25 list (None if absent)
    vector_rank: int | None = None  # 1-based rank in vector list (None if absent)
    document: dict[str, Any] = field(default_factory=dict)


class HybridRanker:
    """Merges BM25 and vector ranked lists via Reciprocal Rank Fusion.

    Parameters
    ----------
    k:
        RRF smoothing constant (default 60, per Cormack et al. 2009).
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        bm25_results: list[tuple[str, float]],   # [(doc_id, score), ...]
        vector_results: list[tuple[str, float]],  # [(doc_id, score), ...]
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> list[HybridResult]:
        """Fuse two ranked lists and return up to *top_k* results.

        Parameters
        ----------
        bm25_results:
            Ranked list from BM25 (already sorted, descending score).
        vector_results:
            Ranked list from vector search (already sorted, descending score).
        alpha:
            Blend weight: 0.0 = pure BM25, 1.0 = pure vector.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[HybridResult] sorted by descending RRF score.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        bm25_weight = 1.0 - alpha
        vector_weight = alpha

        # Build rank maps: doc_id → 1-based rank
        bm25_ranks: dict[str, int] = {doc_id: i + 1 for i, (doc_id, _) in enumerate(bm25_results)}
        vector_ranks: dict[str, int] = {doc_id: i + 1 for i, (doc_id, _) in enumerate(vector_results)}

        # Union of all doc IDs
        all_ids: set[str] = set(bm25_ranks) | set(vector_ranks)

        k = self.k
        scores: dict[str, float] = {}
        for doc_id in all_ids:
            score = 0.0
            if bm25_weight > 0 and doc_id in bm25_ranks:
                score += bm25_weight / (k + bm25_ranks[doc_id])
            if vector_weight > 0 and doc_id in vector_ranks:
                score += vector_weight / (k + vector_ranks[doc_id])
            scores[doc_id] = score

        sorted_ids = sorted(
            (doc_id for doc_id, s in scores.items() if s > 0),
            key=lambda x: scores[x],
            reverse=True,
        )[:top_k]

        return [
            HybridResult(
                id=doc_id,
                score=scores[doc_id],
                bm25_rank=bm25_ranks.get(doc_id),
                vector_rank=vector_ranks.get(doc_id),
            )
            for doc_id in sorted_ids
        ]
