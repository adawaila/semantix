"""Abstract base class for embedding providers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract embedding provider.

    All implementations must be thread-safe — embed_one / embed_batch may
    be called concurrently from multiple ingestion workers.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of produced embeddings."""

    @abstractmethod
    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string.

        Returns
        -------
        np.ndarray of shape (dim,), dtype float32, L2-normalised.
        """

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings.

        Default implementation calls embed_one sequentially.
        Providers should override this with a batched implementation.

        Returns
        -------
        np.ndarray of shape (len(texts), dim), dtype float32.
        """
        return np.stack([self.embed_one(t) for t in texts], axis=0)
