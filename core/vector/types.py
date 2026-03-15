"""Shared types for vectr."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Vector:
    """A stored vector with id and optional metadata."""
    id: str
    data: np.ndarray  # shape (dim,), dtype float32
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 1:
            raise ValueError(f"Vector data must be 1-D, got shape {self.data.shape}")


@dataclass(order=True)
class SearchResult:
    """A single search result."""
    score: float          # cosine similarity, higher = more similar
    id: str = field(compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
