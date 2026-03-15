"""Collection schema and document store.

A Collection owns:
  - A BM25 inverted index for keyword search
  - An HNSW (or BruteForce) vector index for semantic search
  - A raw document store (dict) mapping id → JSON document

Documents are arbitrary dicts with a required "id" field (string).
The embedding_field specifies which field's text is indexed in the vector index.

Persistence uses atomic pickle: write to .tmp then os.rename.
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .bm25 import BM25Index
from .hybrid import HybridRanker, HybridResult
from .vector import HNSWIndex, BruteForceIndex, Vector


class IndexType(str, Enum):
    HNSW = "hnsw"
    BRUTE_FORCE = "brute_force"


@dataclass
class CollectionConfig:
    """Static configuration for a collection."""
    name: str
    embedding_field: str            # which document field to embed
    embedding_dim: int = 384        # dimensionality of vectors
    index_type: IndexType = IndexType.HNSW
    provider: str = "local"         # "local" or "openai"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50


@dataclass
class SearchResult:
    """Result returned from Collection.search()."""
    id: str
    score: float
    document: dict[str, Any]
    bm25_rank: int | None = None
    vector_rank: int | None = None


class Collection:
    """Thread-safe collection combining BM25 + vector search.

    Parameters
    ----------
    config:
        Static collection configuration.
    data_dir:
        Optional directory path for persistence. If None, in-memory only.
    """

    def __init__(self, config: CollectionConfig, data_dir: str | None = None) -> None:
        self.config = config
        self.data_dir = data_dir
        self._lock = threading.RLock()

        # Raw document store: id → dict
        self._docs: dict[str, dict[str, Any]] = {}

        # BM25 inverted index
        self._bm25 = BM25Index()

        # Vector index
        if config.index_type == IndexType.HNSW:
            self._vec_index: HNSWIndex | BruteForceIndex = HNSWIndex(
                dim=config.embedding_dim,
                M=config.hnsw_m,
                ef_construction=config.hnsw_ef_construction,
                ef_search=config.hnsw_ef_search,
            )
        else:
            self._vec_index = BruteForceIndex(dim=config.embedding_dim)

        # Hybrid ranker
        self._ranker = HybridRanker()

        # Stats
        self._created_at: float = time.time()
        self._query_count: int = 0

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def add(self, document: dict[str, Any], embedding: np.ndarray | None = None) -> None:
        """Add or replace a document.

        Parameters
        ----------
        document:
            Must contain an "id" field (string).
        embedding:
            Pre-computed embedding for the embedding_field. If None the vector
            index is not updated (BM25 only).
        """
        doc_id = str(document.get("id", ""))
        if not doc_id:
            raise ValueError("Document must have a non-empty 'id' field")

        text = str(document.get(self.config.embedding_field, ""))

        with self._lock:
            self._docs[doc_id] = dict(document)
            self._bm25.add(doc_id, text)

            if embedding is not None:
                if embedding.shape[0] != self.config.embedding_dim:
                    raise ValueError(
                        f"Embedding dim mismatch: expected {self.config.embedding_dim}, "
                        f"got {embedding.shape[0]}"
                    )
                self._vec_index.add(Vector(id=doc_id, data=embedding))

    def add_batch(
        self,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Batch add documents.

        Parameters
        ----------
        documents:
            List of document dicts, each with "id".
        embeddings:
            2-D array of shape (len(documents), embedding_dim). Optional.
        """
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError("embeddings length must match documents length")

        for i, doc in enumerate(documents):
            emb = embeddings[i] if embeddings is not None else None
            self.add(doc, emb)

    def delete(self, doc_id: str) -> bool:
        """Delete document by ID. Returns True if it existed."""
        with self._lock:
            if doc_id not in self._docs:
                return False
            self._docs.pop(doc_id)
            self._bm25.remove(doc_id)
            self._vec_index.remove(doc_id)
            return True

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """Return the raw document dict or None."""
        with self._lock:
            doc = self._docs.get(doc_id)
            return dict(doc) if doc else None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining BM25 and vector results via RRF.

        Parameters
        ----------
        query_text:
            Text query for BM25.
        query_embedding:
            Embedding for vector search. If None, alpha is forced to 0 (pure BM25).
        top_k:
            Maximum results to return.
        alpha:
            0.0 = pure BM25, 1.0 = pure vector.
        filters:
            Exact-match filters on document fields, e.g. {"category": "electronics"}.
        """
        with self._lock:
            self._query_count += 1

            # BM25 search
            bm25_raw = self._bm25.search(query_text, top_k=top_k * 2) if alpha < 1.0 else []

            # Vector search
            vec_raw: list[tuple[str, float]] = []
            if query_embedding is not None and alpha > 0.0:
                vec_results = self._vec_index.search(query_embedding, top_k=top_k * 2)
                vec_raw = [(r.id, r.score) for r in vec_results]

            # If no embedding provided, force pure BM25
            effective_alpha = 0.0 if query_embedding is None else alpha

            # Fuse
            fused: list[HybridResult] = self._ranker.fuse(
                bm25_raw, vec_raw, alpha=effective_alpha, top_k=top_k * 2
            )

            # Attach documents and apply filters
            results: list[SearchResult] = []
            for hr in fused:
                doc = self._docs.get(hr.id)
                if doc is None:
                    continue
                if filters and not self._matches_filters(doc, filters):
                    continue
                results.append(SearchResult(
                    id=hr.id,
                    score=hr.score,
                    document=dict(doc),
                    bm25_rank=hr.bm25_rank,
                    vector_rank=hr.vector_rank,
                ))
                if len(results) == top_k:
                    break

            return results

    @staticmethod
    def _matches_filters(doc: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Return True if doc satisfies all filter conditions.

        Supports:
          - Exact match:    {"category": "electronics"}
          - Range:          {"price": {"$gte": 10, "$lte": 50}}
          - List contains:  {"tags": {"$contains": "sale"}}
        """
        for field_name, condition in filters.items():
            val = doc.get(field_name)
            if isinstance(condition, dict):
                for op, operand in condition.items():
                    if op == "$gte" and not (val is not None and val >= operand):
                        return False
                    elif op == "$lte" and not (val is not None and val <= operand):
                        return False
                    elif op == "$gt" and not (val is not None and val > operand):
                        return False
                    elif op == "$lt" and not (val is not None and val < operand):
                        return False
                    elif op == "$contains" and not (
                        isinstance(val, (list, str)) and operand in val
                    ):
                        return False
                    elif op == "$ne" and val == operand:
                        return False
            else:
                # Exact match
                if val != condition:
                    return False
        return True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        with self._lock:
            return {
                "name": self.config.name,
                "doc_count": len(self._docs),
                "vocab_size": self._bm25.vocab_size(),
                "vector_count": len(self._vec_index),
                "embedding_dim": self.config.embedding_dim,
                "embedding_field": self.config.embedding_field,
                "index_type": self.config.index_type.value,
                "provider": self.config.provider,
                "query_count": self._query_count,
                "created_at": self._created_at,
            }

    def list_docs(self, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Return a paginated slice of documents and the total count."""
        with self._lock:
            all_docs = list(self._docs.values())
        total = len(all_docs)
        page = [dict(d) for d in all_docs[offset : offset + limit]]
        return page, total

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _data_path(self) -> str | None:
        if self.data_dir is None:
            return None
        return os.path.join(self.data_dir, f"{self.config.name}.pkl")

    def save(self) -> None:
        """Atomically persist collection to disk."""
        path = self._data_path()
        if path is None:
            return

        with self._lock:
            state = {
                "config": self.config,
                "docs": dict(self._docs),
                "bm25_state": {
                    "docs": dict(self._bm25._docs),
                    "doc_tokens": dict(self._bm25._doc_tokens),
                    "doc_len": dict(self._bm25._doc_len),
                    "index": {k: dict(v) for k, v in self._bm25._index.items()},
                    "total_len": self._bm25._total_len,
                },
                "vec_state": self._vec_index.state(),
                "is_hnsw": isinstance(self._vec_index, HNSWIndex),
                "query_count": self._query_count,
                "created_at": self._created_at,
            }

        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Collection":
        """Load a collection from a pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        config: CollectionConfig = state["config"]
        col = cls(config, data_dir=os.path.dirname(path))

        # Restore docs
        col._docs = state["docs"]

        # Restore BM25
        bs = state["bm25_state"]
        col._bm25._docs = bs["docs"]
        col._bm25._doc_tokens = bs["doc_tokens"]
        col._bm25._doc_len = bs["doc_len"]
        col._bm25._index = bs["index"]
        col._bm25._total_len = bs["total_len"]

        # Restore vector index
        if state["is_hnsw"]:
            col._vec_index = HNSWIndex.from_state(state["vec_state"])
        else:
            col._vec_index = BruteForceIndex.from_state(state["vec_state"])

        col._query_count = state.get("query_count", 0)
        col._created_at = state.get("created_at", time.time())
        return col


class CollectionStore:
    """Thread-safe registry of all collections.

    Optionally persists each collection to data_dir/<name>.pkl.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        self.data_dir = data_dir
        self._lock = threading.RLock()
        self._collections: dict[str, Collection] = {}

        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.data_dir:
            return
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".pkl"):
                try:
                    col = Collection.load(os.path.join(self.data_dir, fname))
                    self._collections[col.config.name] = col
                except Exception:
                    pass  # corrupt file — skip

    def create(self, config: CollectionConfig) -> Collection:
        """Create a new collection. Raises ValueError if name already exists."""
        with self._lock:
            if config.name in self._collections:
                raise ValueError(f"Collection '{config.name}' already exists")
            col = Collection(config, data_dir=self.data_dir)
            self._collections[config.name] = col
            if self.data_dir:
                col.save()
            return col

    def get(self, name: str) -> Collection | None:
        with self._lock:
            return self._collections.get(name)

    def list(self) -> list[str]:
        with self._lock:
            return list(self._collections)

    def delete(self, name: str) -> bool:
        with self._lock:
            if name not in self._collections:
                return False
            del self._collections[name]
            if self.data_dir:
                path = os.path.join(self.data_dir, f"{name}.pkl")
                if os.path.exists(path):
                    os.remove(path)
            return True

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._collections
