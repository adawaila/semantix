"""Pydantic v2 request/response models for the semantix API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    embedding_field: str = Field(default="description")
    embedding_dim: int = Field(default=384, ge=1, le=8192)
    index_type: Literal["hnsw", "brute_force"] = "hnsw"
    provider: Literal["local", "openai"] = "local"
    hnsw_m: int = Field(default=16, ge=4, le=64)
    hnsw_ef_construction: int = Field(default=200, ge=16, le=1000)
    hnsw_ef_search: int = Field(default=50, ge=10, le=500)


class CollectionStats(BaseModel):
    name: str
    doc_count: int
    vocab_size: int
    vector_count: int
    embedding_dim: int
    embedding_field: str = "description"
    index_type: str
    provider: str
    query_count: int
    created_at: float


class CollectionListResponse(BaseModel):
    collections: list[CollectionStats]
    total: int


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class IngestDocumentRequest(BaseModel):
    document: dict[str, Any]

    @field_validator("document")
    @classmethod
    def must_have_id(cls, v: dict) -> dict:
        if "id" not in v or not str(v["id"]).strip():
            raise ValueError("Document must have a non-empty 'id' field")
        return v


class BulkIngestRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(..., min_length=1)

    @field_validator("documents")
    @classmethod
    def all_must_have_id(cls, v: list[dict]) -> list[dict]:
        for i, doc in enumerate(v):
            if "id" not in doc or not str(doc["id"]).strip():
                raise ValueError(f"Document at index {i} must have a non-empty 'id' field")
        return v


class IngestResponse(BaseModel):
    indexed: int
    errors: int


class BulkIngestResponse(BaseModel):
    job_id: str
    total: int
    status: str = "queued"


class DocumentListResponse(BaseModel):
    documents: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=1000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    filters: dict[str, Any] | None = None


class SearchResultItem(BaseModel):
    id: str
    score: float
    document: dict[str, Any]
    bm25_rank: int | None = None
    vector_rank: int | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total_docs: int
    latency_ms: float
    query: str


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    job_id: str
    status: str
    total: int
    done: int
    errors: int
    error_msg: str
    progress: float


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
