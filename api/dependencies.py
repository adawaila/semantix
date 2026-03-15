"""Shared FastAPI dependencies."""
from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, HTTPException

from core.collection import CollectionStore
from embeddings import get_provider
from embeddings.base import EmbeddingProvider
from ingestion.jobs import JobTracker

# Singletons (initialised in app.py lifespan)
_store: CollectionStore | None = None
_tracker: JobTracker | None = None
_provider: EmbeddingProvider | None = None
_redis = None


def init_dependencies(
    store: CollectionStore,
    tracker: JobTracker | None,
    provider: EmbeddingProvider,
    redis_client=None,
) -> None:
    global _store, _tracker, _provider, _redis
    _store = store
    _tracker = tracker
    _provider = provider
    _redis = redis_client


def get_store() -> CollectionStore:
    assert _store is not None, "store not initialised"
    return _store


def get_tracker() -> JobTracker | None:
    return _tracker


def get_provider_dep() -> EmbeddingProvider:
    assert _provider is not None, "provider not initialised"
    return _provider


def get_redis():
    return _redis


def get_collection(name: str, store: CollectionStore = Depends(get_store)):
    col = store.get(name)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return col
