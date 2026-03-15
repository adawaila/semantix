"""Brute-force cosine similarity index.

Stores all vectors in a matrix and computes exact cosine similarity
against every vector at query time.  O(n*d) per query but 100% recall.

Performance notes
-----------------
- add() is O(d) — unit vectors are appended to a list; existence checks
  use an O(1) dict (_id_to_pos) instead of a linear list scan.
- add_batch() normalises outside the lock, then does one O(1) dict
  lookup per vector — total O(n*d) for n vectors.
- The NumPy matrix is rebuilt eagerly at the end of each write (inside the
  write lock), so search() holds only a read lock and never mutates state.
- search() is O(n*d) — one BLAS matrix-vector multiply.
- argpartition gives O(n) top-k selection instead of a full sort.
"""
from __future__ import annotations
from typing import Any

import numpy as np

from .types import Vector, SearchResult
from .rwlock import RWLock


class BruteForceIndex:
    """Exact nearest-neighbour search via cosine similarity.

    Thread-safe: concurrent reads (search/get/ids) run in parallel;
    writes (add/remove) are exclusive via a writer-preferring RWLock.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self._lock = RWLock()

        self._ids: list[str] = []                    # ordered list of string IDs
        self._id_to_pos: dict[str, int] = {}         # O(1) id → position lookup
        self._meta: list[dict[str, Any]] = []
        self._units: list[np.ndarray] = []           # unit-normed float32 (dim,)
        self._matrix: np.ndarray = np.empty((0, dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _rebuild(self) -> None:
        """Rebuild the vector matrix after a mutation. Called under write lock."""
        self._matrix = (
            np.stack(self._units, axis=0)
            if self._units
            else np.empty((0, self.dim), dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, vector: Vector) -> None:
        """Insert or update *vector*.  O(d)."""
        if vector.data.shape[0] != self.dim:
            raise ValueError(
                f"Dimension mismatch: index dim={self.dim}, "
                f"vector dim={vector.data.shape[0]}"
            )
        unit = self._normalise(vector.data)

        with self._lock.write():
            if vector.id in self._id_to_pos:          # O(1) dict lookup
                pos = self._id_to_pos[vector.id]
                self._units[pos] = unit
                self._meta[pos] = vector.metadata
            else:
                pos = len(self._ids)
                self._id_to_pos[vector.id] = pos
                self._ids.append(vector.id)
                self._meta.append(vector.metadata)
                self._units.append(unit)
            self._rebuild()

    def add_batch(self, vectors: list[Vector]) -> None:
        """Bulk insert.  Validates all dims, normalises outside the lock."""
        for v in vectors:
            if v.data.shape[0] != self.dim:
                raise ValueError(
                    f"Dimension mismatch: index dim={self.dim}, "
                    f"vector '{v.id}' dim={v.data.shape[0]}"
                )
        normed = [self._normalise(v.data) for v in vectors]   # outside lock

        with self._lock.write():
            for v, unit in zip(vectors, normed):
                if v.id in self._id_to_pos:                   # O(1)
                    pos = self._id_to_pos[v.id]
                    self._units[pos] = unit
                    self._meta[pos] = v.metadata
                else:
                    pos = len(self._ids)
                    self._id_to_pos[v.id] = pos
                    self._ids.append(v.id)
                    self._meta.append(v.metadata)
                    self._units.append(unit)
            self._rebuild()

    def remove(self, vector_id: str) -> bool:
        """Remove by id.  O(n) due to position-shift, rare in practice."""
        with self._lock.write():
            if vector_id not in self._id_to_pos:
                return False
            pos = self._id_to_pos.pop(vector_id)
            self._ids.pop(pos)
            self._meta.pop(pos)
            self._units.pop(pos)
            # Shift positions of everything after the removed slot
            for i in range(pos, len(self._ids)):
                self._id_to_pos[self._ids[i]] = i
            self._rebuild()
            return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Return *top_k* results by cosine similarity, descending."""
        if query.shape[0] != self.dim:
            raise ValueError(
                f"Dimension mismatch: index dim={self.dim}, "
                f"query dim={query.shape[0]}"
            )
        with self._lock.read():
            n = len(self._ids)
            if n == 0:
                return []
            top_k = min(top_k, n)

            q = query.astype(np.float32)
            qnorm = np.linalg.norm(q)
            if qnorm > 0:
                q = q / qnorm

            scores: np.ndarray = self._matrix @ q   # O(n*d) BLAS

            if top_k == n:
                idxs = np.argsort(scores)[::-1]
            else:
                part = np.argpartition(scores, -top_k)[-top_k:]
                idxs = part[np.argsort(scores[part])[::-1]]

            return [
                SearchResult(
                    score=float(scores[i]),
                    id=self._ids[i],
                    metadata=dict(self._meta[i]),
                )
                for i in idxs
            ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock.read():
            return len(self._ids)

    @property
    def size(self) -> int:
        return len(self)

    def get(self, vector_id: str) -> Vector | None:
        with self._lock.read():
            if vector_id not in self._id_to_pos:
                return None
            pos = self._id_to_pos[vector_id]
            return Vector(
                id=vector_id,
                data=self._units[pos].copy(),
                metadata=dict(self._meta[pos]),
            )

    def ids(self) -> list[str]:
        with self._lock.read():
            return list(self._ids)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state(self) -> dict:
        with self._lock.read():
            return {
                "dim": self.dim,
                "ids": list(self._ids),
                "meta": list(self._meta),
                "matrix": self._matrix.copy(),
            }

    @classmethod
    def from_state(cls, state: dict) -> "BruteForceIndex":
        obj = cls(state["dim"])
        obj._ids = state["ids"]
        obj._id_to_pos = {sid: i for i, sid in enumerate(obj._ids)}
        obj._meta = state["meta"]
        mat: np.ndarray = state["matrix"]
        obj._units = [mat[i] for i in range(len(mat))]
        obj._matrix = mat.copy()
        return obj
