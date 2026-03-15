"""SDK-side Pydantic models."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class CollectionStats(BaseModel):
    name: str
    doc_count: int
    vocab_size: int
    vector_count: int
    embedding_dim: int
    index_type: str
    provider: str
    query_count: int
    created_at: float


class SearchResult(BaseModel):
    id: str
    score: float
    document: dict[str, Any]
    bm25_rank: int | None = None
    vector_rank: int | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_docs: int
    latency_ms: float
    query: str


class IngestResponse(BaseModel):
    indexed: int
    errors: int


class BulkIngestResponse(BaseModel):
    job_id: str
    total: int
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    total: int
    done: int
    errors: int
    error_msg: str
    progress: float
