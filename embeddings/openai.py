"""OpenAI embedding provider (text-embedding-3-small).

Requires OPENAI_API_KEY environment variable.
Uses tenacity for automatic retry with exponential backoff.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .base import EmbeddingProvider

_DIM = 1536  # text-embedding-3-small output dimension


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI text-embedding-3-small provider with batching and retry.

    Parameters
    ----------
    api_key:
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    model:
        OpenAI embedding model name.
    batch_size:
        Number of texts per API call (max 2048 per OpenAI limits).
    max_retries:
        Maximum retry attempts on transient errors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @property
    def dim(self) -> int:
        return _DIM

    def _embed_chunk(self, texts: list[str]) -> np.ndarray:
        """Embed a single chunk with retry."""
        client = self._get_client()
        for attempt in range(self._max_retries):
            try:
                response = client.embeddings.create(input=texts, model=self._model)
                vecs = [item.embedding for item in response.data]
                arr = np.array(vecs, dtype=np.float32)
                # Normalise
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                return arr / norms
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def embed_one(self, text: str) -> np.ndarray:
        return self._embed_chunk([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        chunks = [
            texts[i: i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
        return np.concatenate([self._embed_chunk(chunk) for chunk in chunks], axis=0)
