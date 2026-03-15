"""Ingestion pipeline: chunk → embed → index.

IngestionPipeline.ingest() is the main synchronous entry point used both by
the API (for single-document immediate indexing) and by the worker process
(for batch async indexing).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.collection import Collection
from embeddings.base import EmbeddingProvider
from .chunker import Chunker

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates chunking, embedding, and indexing for a collection.

    Parameters
    ----------
    collection:
        The target Collection.
    provider:
        Embedding provider.
    chunk_size:
        Characters per chunk (0 disables chunking).
    chunk_overlap:
        Overlap between consecutive chunks.
    embed_batch_size:
        Number of chunks per embedding call.
    """

    def __init__(
        self,
        collection: Collection,
        provider: EmbeddingProvider,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embed_batch_size: int = 32,
    ) -> None:
        self.collection = collection
        self.provider = provider
        self.embed_batch_size = embed_batch_size

        if chunk_size > 0:
            self._chunker: Chunker | None = Chunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_field=collection.config.embedding_field,
            )
        else:
            self._chunker = None

    def ingest(self, documents: list[dict[str, Any]]) -> dict[str, int]:
        """Ingest a list of documents.

        Returns a dict with keys "indexed" and "errors".
        """
        indexed = 0
        errors = 0

        # Step 1: chunk
        all_chunks: list[dict[str, Any]] = []
        for doc in documents:
            try:
                if self._chunker is not None:
                    chunks = self._chunker.chunk_document(doc)
                    all_chunks.extend(c.document for c in chunks)
                else:
                    all_chunks.append(doc)
            except Exception as e:
                logger.warning("Chunking failed for doc %s: %s", doc.get("id"), e)
                errors += 1

        if not all_chunks:
            return {"indexed": 0, "errors": errors}

        # Step 2: embed in batches
        texts = [
            str(c.get(self.collection.config.embedding_field, ""))
            for c in all_chunks
        ]
        embeddings: list[np.ndarray | None] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch_texts = texts[i: i + self.embed_batch_size]
            try:
                batch_embs = self.provider.embed_batch(batch_texts)
                embeddings.extend(batch_embs)
            except Exception as e:
                logger.error("Embedding batch %d failed: %s", i // self.embed_batch_size, e)
                embeddings.extend([None] * len(batch_texts))
                errors += len(batch_texts)

        # Step 3: index
        for chunk_doc, emb in zip(all_chunks, embeddings):
            try:
                self.collection.add(chunk_doc, embedding=emb)
                indexed += 1
            except Exception as e:
                logger.warning("Indexing failed for %s: %s", chunk_doc.get("id"), e)
                errors += 1

        return {"indexed": indexed, "errors": errors}

    def ingest_one(self, document: dict[str, Any]) -> None:
        """Ingest a single document (synchronous, immediate)."""
        self.ingest([document])
