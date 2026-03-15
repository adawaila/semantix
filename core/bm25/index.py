"""Okapi BM25 inverted index.

Implements BM25 ranking (Robertson & Ogilvie, 1994) with:
  - k1 = 1.5   (term frequency saturation)
  - b  = 0.75  (document length normalisation)

All term lookups are O(1) via dict-based inverted index.
Thread-safe: multiple readers, exclusive writers via threading.RLock.
"""
from __future__ import annotations

import math
import threading
from typing import Any

from .tokenizer import Tokenizer


class BM25Index:
    """BM25 inverted index over arbitrary text documents.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter (default 1.5).
    b:
        Length normalisation parameter (default 0.75).
    use_stemming:
        Whether to stem tokens during indexing/querying.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_stemming: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._tokenizer = Tokenizer(use_stemming=use_stemming)
        self._lock = threading.RLock()

        # doc_id → raw text (for term-freq recomputation on delete)
        self._docs: dict[str, str] = {}
        # doc_id → token list
        self._doc_tokens: dict[str, list[str]] = {}
        # doc_id → token count (document length)
        self._doc_len: dict[str, int] = {}

        # Inverted index: term → {doc_id → term frequency}
        self._index: dict[str, dict[str, int]] = {}

        # Running sum of all document lengths (for avg_dl)
        self._total_len: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _n_docs(self) -> int:
        return len(self._docs)

    @property
    def _avg_dl(self) -> float:
        if not self._docs:
            return 0.0
        return self._total_len / len(self._docs)

    def _idf(self, term: str) -> float:
        """Robertson's IDF with smoothing to prevent negative values."""
        df = len(self._index.get(term, {}))
        n = self._n_docs
        if df == 0 or n == 0:
            return 0.0
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, doc_id: str, text: str) -> None:
        """Index *text* under *doc_id*.  Replaces any existing entry."""
        tokens = self._tokenizer.tokenize(text)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        with self._lock:
            # Remove old entry if present
            if doc_id in self._docs:
                self._remove_no_lock(doc_id)

            self._docs[doc_id] = text
            self._doc_tokens[doc_id] = tokens
            self._doc_len[doc_id] = len(tokens)
            self._total_len += len(tokens)

            for term, freq in tf.items():
                if term not in self._index:
                    self._index[term] = {}
                self._index[term][doc_id] = freq

    def remove(self, doc_id: str) -> bool:
        """Remove *doc_id* from the index.  Returns True if it existed."""
        with self._lock:
            return self._remove_no_lock(doc_id)

    def _remove_no_lock(self, doc_id: str) -> bool:
        if doc_id not in self._docs:
            return False
        tokens = self._doc_tokens.pop(doc_id)
        self._total_len -= self._doc_len.pop(doc_id)
        self._docs.pop(doc_id)

        seen: set[str] = set()
        for t in tokens:
            if t not in seen:
                seen.add(t)
                postings = self._index.get(t)
                if postings is not None:
                    postings.pop(doc_id, None)
                    if not postings:
                        del self._index[t]
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to *top_k* (doc_id, score) pairs, descending BM25 score.

        Returns an empty list for an empty index or a query with no hits.
        """
        q_tokens = self._tokenizer.tokenize(query)
        if not q_tokens:
            return []

        with self._lock:
            n = self._n_docs
            avg_dl = self._avg_dl
            if n == 0:
                return []

            scores: dict[str, float] = {}
            k1, b = self.k1, self.b

            for term in set(q_tokens):  # unique terms only
                postings = self._index.get(term)
                if not postings:
                    continue
                idf = self._idf(term)
                for doc_id, tf in postings.items():
                    dl = self._doc_len[doc_id]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * dl / avg_dl)
                    scores[doc_id] = scores.get(doc_id, 0.0) + idf * numerator / denominator

            if not scores:
                return []

            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_docs[:top_k]

    def term_freq(self, doc_id: str, term: str) -> int:
        """Return the raw term frequency of *term* in *doc_id*."""
        with self._lock:
            stemmed = self._tokenizer.tokenize(term)
            if not stemmed:
                return 0
            return self._index.get(stemmed[0], {}).get(doc_id, 0)

    def doc_count(self) -> int:
        """Return the number of indexed documents."""
        with self._lock:
            return self._n_docs

    def vocab_size(self) -> int:
        """Return the number of unique terms in the index."""
        with self._lock:
            return len(self._index)

    def contains(self, doc_id: str) -> bool:
        """Return True if *doc_id* is in the index."""
        with self._lock:
            return doc_id in self._docs

    def ids(self) -> list[str]:
        """Return all indexed document IDs."""
        with self._lock:
            return list(self._docs)
