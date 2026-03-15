"""Local embedding provider using sentence-transformers (fully offline).

Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality).
The model is downloaded once to the HuggingFace cache on first use.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from .base import EmbeddingProvider


class LocalEmbeddings(EmbeddingProvider):
    """sentence-transformers embeddings, runs fully offline after first download.

    Parameters
    ----------
    model_name:
        HuggingFace model name (default: "sentence-transformers/all-MiniLM-L6-v2").
    batch_size:
        Batch size for embed_batch (default 32).
    device:
        "cpu", "cuda", or "mps". If None, auto-detected.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._lock = threading.Lock()
        self._model = None  # lazy load

    def _load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(
                        self._model_name,
                        device=self._device,
                    )

    @property
    def dim(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def embed_one(self, text: str) -> np.ndarray:
        self._load_model()
        emb = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1,
        )
        return emb[0].astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        self._load_model()
        embs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )
        return embs.astype(np.float32)
