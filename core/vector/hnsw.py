"""HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.

Pure-Python implementation following the original paper:
  Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
  using Hierarchical Navigable Small World graphs", IEEE TPAMI 2020.

Only numpy is used for vector arithmetic; no third-party ANN libraries.

Performance design
------------------
All internal hot-path data structures use integer indices rather than string
IDs to minimise Python dict overhead and enable fast numpy batch-distance
computation:

  _vecs:  numpy array of shape (n, dim), each row is a unit-normed vector.
  _neighbors[i][layer]: list[int] of neighbour indices at that layer.

String IDs are mapped ↔ integer indices via _id_to_idx / _idx_to_id.  The
hot-paths (search, connect) never touch Python dicts.
"""
from __future__ import annotations

import heapq
import math
import random
from typing import Any

import numpy as np

from .types import Vector, SearchResult
from .rwlock import RWLock


# ---------------------------------------------------------------------------
# HNSW Index
# ---------------------------------------------------------------------------

class HNSWIndex:
    """Approximate nearest-neighbour search via HNSW.

    Parameters
    ----------
    dim:
        Dimensionality of stored vectors.
    M:
        Max bidirectional links per node per layer (default 16).
    ef_construction:
        Candidate list size during construction (default 200).
    ef_search:
        Candidate list size during search (default 50).
    ml:
        Level generation multiplier.  Default 1/ln(M).
    seed:
        RNG seed for reproducible level sampling.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: float | None = None,
        seed: int | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml if ml is not None else (1.0 / math.log(M) if M > 1 else 1.0)
        self._rng = random.Random(seed)
        self._lock = RWLock()

        # ---- Storage (parallel arrays, index i = node i) ---------------
        # Unit-normed float32 vectors; we grow this with np.concatenate.
        self._vecs: np.ndarray = np.empty((0, dim), dtype=np.float32)
        # Python lists (one per node): neighbors[i][layer] = list[int idx]
        self._neighbors: list[list[list[int]]] = []
        self._metadata: list[dict[str, Any]] = []

        # String ID ↔ integer index maps
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: list[str] = []

        # Slot recycling (after remove())
        self._free: list[int] = []   # recycled indices

        self._entry_idx: int = -1    # index of current entry node (-1 = empty)
        self._max_layer: int = 0

    # ------------------------------------------------------------------
    # Level sampling
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random()) * self.ml)

    # ------------------------------------------------------------------
    # Core HNSW search
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
        entry_idxs: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Greedy beam search in *layer*.  Returns (dist, idx) list, ascending."""
        visited: set[int] = set(entry_idxs)
        # min-heap: (dist, idx) — pop nearest candidate first
        cands: list[tuple[float, int]] = []
        # max-heap via negation: (-dist, idx) — pop farthest result to evict
        W: list[tuple[float, int]] = []

        # Seed with entry points
        for ei in entry_idxs:
            d = float(1.0 - float(np.dot(self._vecs[ei], query)))
            heapq.heappush(cands, (d, ei))
            heapq.heappush(W, (-d, ei))

        while cands:
            c_dist, c_idx = heapq.heappop(cands)
            worst = -W[0][0] if W else float("inf")
            if c_dist > worst:
                break

            nb_list = self._neighbors[c_idx]
            layer_nb: list[int] = nb_list[layer] if layer < len(nb_list) else []

            # Collect unvisited neighbours
            new_nbs: list[int] = []
            for nb in layer_nb:
                if nb not in visited:
                    visited.add(nb)
                    new_nbs.append(nb)

            if not new_nbs:
                continue

            # Batch dot-product: all unvisited neighbours vs query in one call
            nb_vecs = self._vecs[new_nbs]        # (k, dim)
            dots = nb_vecs @ query               # (k,)

            worst = -W[0][0] if W else float("inf")
            for nb_idx, dot in zip(new_nbs, dots.tolist()):
                nb_dist = 1.0 - dot
                if nb_dist < worst or len(W) < ef:
                    heapq.heappush(cands, (nb_dist, nb_idx))
                    heapq.heappush(W, (-nb_dist, nb_idx))
                    if len(W) > ef:
                        heapq.heappop(W)
                        worst = -W[0][0]

        return sorted((-d, i) for d, i in W)

    def _select_heuristic(
        self,
        query: np.ndarray,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """Return at most M diverse neighbour indices (HNSW heuristic)."""
        if len(candidates) <= M:
            return [i for _, i in candidates]

        kept: list[tuple[float, int]] = []
        discarded: list[tuple[float, int]] = []

        for dist, idx in candidates:
            if len(kept) >= M:
                break
            good = True
            for _, r_idx in kept:
                if float(1.0 - float(np.dot(self._vecs[idx], self._vecs[r_idx]))) < dist:
                    good = False
                    break
            if good:
                kept.append((dist, idx))
            else:
                discarded.append((dist, idx))

        for dist, idx in discarded:
            if len(kept) >= M:
                break
            kept.append((dist, idx))

        return [i for _, i in kept]

    def _connect(self, new_idx: int, neighbour_idxs: list[int], layer: int) -> None:
        """Add bidirectional edges; shrink lists that exceed M_max."""
        M_max = self.M_max0 if layer == 0 else self.M
        nb_list = self._neighbors[new_idx]
        while len(nb_list) <= layer:
            nb_list.append([])
        nb_list[layer] = neighbour_idxs

        for nb_idx in neighbour_idxs:
            nb_nb_list = self._neighbors[nb_idx]
            while len(nb_nb_list) <= layer:
                nb_nb_list.append([])
            nb_nb_list[layer].append(new_idx)

            if len(nb_nb_list[layer]) > M_max:
                vecs = self._vecs[[nb_idx]]   # shape (1, dim)
                cands = sorted(
                    (float(1.0 - float(np.dot(self._vecs[c], self._vecs[nb_idx]))), c)
                    for c in nb_nb_list[layer]
                )
                nb_nb_list[layer] = self._select_heuristic(
                    self._vecs[nb_idx], cands, M_max
                )

    # ------------------------------------------------------------------
    # Allocation helpers
    # ------------------------------------------------------------------

    def _alloc_idx(self) -> int:
        """Return a fresh integer index (recycling freed slots)."""
        if self._free:
            return self._free.pop()
        return len(self._idx_to_id)

    def _store_node(
        self, idx: int, sid: str, unit: np.ndarray, meta: dict, level: int
    ) -> None:
        """Write node data into parallel storage at position *idx*."""
        n = self._vecs.shape[0]
        if idx < n:
            self._vecs[idx] = unit
            self._neighbors[idx] = [[] for _ in range(level + 1)]
            self._metadata[idx] = meta
            self._idx_to_id[idx] = sid
        else:
            # Append new row
            self._vecs = np.concatenate(
                [self._vecs, unit[np.newaxis, :]], axis=0
            )
            self._neighbors.append([[] for _ in range(level + 1)])
            self._metadata.append(meta)
            self._idx_to_id.append(sid)

    # ------------------------------------------------------------------
    # Public mutation
    # ------------------------------------------------------------------

    def add(self, vector: Vector) -> None:
        if vector.data.shape[0] != self.dim:
            raise ValueError(
                f"Dimension mismatch: index dim={self.dim}, "
                f"vector dim={vector.data.shape[0]}"
            )
        raw = vector.data.astype(np.float32)
        norm = np.linalg.norm(raw)
        unit = raw / norm if norm > 0 else raw.copy()

        with self._lock.write():
            if vector.id in self._id_to_idx:
                self._remove_no_lock(vector.id)

            level = self._random_level()
            idx = self._alloc_idx()
            self._store_node(idx, vector.id, unit, vector.metadata, level)
            self._id_to_idx[vector.id] = idx

            if self._entry_idx == -1:
                self._entry_idx = idx
                self._max_layer = level
                return

            ep = [self._entry_idx]
            cur_top = self._max_layer

            # Phase 1: descend from top layer to level+1 (ef=1)
            for lc in range(cur_top, level, -1):
                results = self._search_layer(unit, ep, ef=1, layer=lc)
                ep = [results[0][1]]

            # Phase 2: insert at each layer from min(level,cur_top) down to 0
            for lc in range(min(level, cur_top), -1, -1):
                M_lc = self.M_max0 if lc == 0 else self.M
                results = self._search_layer(unit, ep, ef=self.ef_construction, layer=lc)
                neighbours = self._select_heuristic(unit, results, M_lc)
                self._connect(idx, neighbours, lc)
                ep = [r[1] for r in results]

            if level > cur_top:
                self._max_layer = level
                self._entry_idx = idx

    def add_batch(self, vectors: list[Vector]) -> None:
        for v in vectors:
            self.add(v)

    def _remove_no_lock(self, vector_id: str) -> bool:
        if vector_id not in self._id_to_idx:
            return False
        idx = self._id_to_idx.pop(vector_id)
        node_nb = self._neighbors[idx]
        for layer, nb_list in enumerate(node_nb):
            for nb_idx in nb_list:
                if nb_idx < len(self._neighbors):
                    nb_nb = self._neighbors[nb_idx]
                    if layer < len(nb_nb) and idx in nb_nb[layer]:
                        nb_nb[layer].remove(idx)
        # Zero out the slot
        self._neighbors[idx] = []
        self._metadata[idx] = {}
        self._idx_to_id[idx] = ""
        self._free.append(idx)

        if self._entry_idx == idx:
            # Find a replacement entry point
            live = [i for i, sid in enumerate(self._idx_to_id) if sid and i not in self._free]
            if live:
                self._entry_idx = live[0]
                self._max_layer = max(len(self._neighbors[i]) - 1 for i in live)
            else:
                self._entry_idx = -1
                self._max_layer = 0
        return True

    def remove(self, vector_id: str) -> bool:
        with self._lock.write():
            return self._remove_no_lock(vector_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        ef: int | None = None,
    ) -> list[SearchResult]:
        if query.shape[0] != self.dim:
            raise ValueError(
                f"Dimension mismatch: index dim={self.dim}, "
                f"query dim={query.shape[0]}"
            )
        ef = max(ef or self.ef_search, top_k)

        with self._lock.read():
            if self._entry_idx == -1:
                return []

            q = query.astype(np.float32)
            qn = np.linalg.norm(q)
            if qn > 0:
                q = q / qn

            ep = [self._entry_idx]
            for lc in range(self._max_layer, 0, -1):
                results = self._search_layer(q, ep, ef=1, layer=lc)
                ep = [results[0][1]]

            results = self._search_layer(q, ep, ef=ef, layer=0)

            out: list[SearchResult] = []
            for dist, idx in results[:top_k]:
                sid = self._idx_to_id[idx]
                if sid:
                    out.append(SearchResult(
                        score=float(1.0 - dist),
                        id=sid,
                        metadata=dict(self._metadata[idx]),
                    ))
            return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock.read():
            return len(self._id_to_idx)

    @property
    def size(self) -> int:
        return len(self)

    def ids(self) -> list[str]:
        with self._lock.read():
            return list(self._id_to_idx)

    def get(self, vector_id: str) -> Vector | None:
        with self._lock.read():
            if vector_id not in self._id_to_idx:
                return None
            idx = self._id_to_idx[vector_id]
            return Vector(
                id=vector_id,
                data=self._vecs[idx].copy(),
                metadata=dict(self._metadata[idx]),
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state(self) -> dict:
        with self._lock.read():
            return {
                "dim": self.dim,
                "M": self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "ml": self.ml,
                "max_layer": self._max_layer,
                "entry_idx": self._entry_idx,
                "vecs": self._vecs.copy(),
                "neighbors": [
                    [list(layer) for layer in node_nb]
                    for node_nb in self._neighbors
                ],
                "metadata": list(self._metadata),
                "id_to_idx": dict(self._id_to_idx),
                "idx_to_id": list(self._idx_to_id),
                "free": list(self._free),
            }

    @classmethod
    def from_state(cls, state: dict) -> "HNSWIndex":
        idx = cls(
            dim=state["dim"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            ef_search=state["ef_search"],
            ml=state["ml"],
        )
        idx._max_layer = state["max_layer"]
        idx._entry_idx = state["entry_idx"]
        idx._vecs = state["vecs"]
        idx._neighbors = [[list(lyr) for lyr in node_nb] for node_nb in state["neighbors"]]
        idx._metadata = list(state["metadata"])
        idx._id_to_idx = dict(state["id_to_idx"])
        idx._idx_to_id = list(state["idx_to_id"])
        idx._free = list(state["free"])
        return idx
