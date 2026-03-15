"""Synchronous semantix Python client."""
from __future__ import annotations

from typing import Any

import httpx

from .models import (
    CollectionStats, SearchResponse, IngestResponse,
    BulkIngestResponse, JobResponse,
)


class SemantixError(Exception):
    """Raised when the API returns an error response."""
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class Semantix:
    """Synchronous semantix client.

    Example
    -------
    >>> client = Semantix(base_url="http://localhost:8000")
    >>> client.create_collection("products", embedding_field="description")
    >>> client.ingest("products", [{"id": "1", "description": "wireless headphones"}])
    >>> results = client.search("products", "wireless headphones", top_k=5)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    def _check(self, resp: httpx.Response) -> dict:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise SemantixError(resp.status_code, detail)
        if resp.status_code == 204:
            return {}
        return resp.json()

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        embedding_field: str = "description",
        embedding_dim: int = 384,
        index_type: str = "hnsw",
        provider: str = "local",
        **kwargs,
    ) -> CollectionStats:
        payload = {
            "name": name,
            "embedding_field": embedding_field,
            "embedding_dim": embedding_dim,
            "index_type": index_type,
            "provider": provider,
            **kwargs,
        }
        data = self._check(self._http.post("/collections", json=payload))
        return CollectionStats(**data)

    def list_collections(self) -> list[CollectionStats]:
        data = self._check(self._http.get("/collections"))
        return [CollectionStats(**c) for c in data["collections"]]

    def get_collection(self, name: str) -> CollectionStats:
        data = self._check(self._http.get(f"/collections/{name}"))
        return CollectionStats(**data)

    def delete_collection(self, name: str) -> None:
        self._check(self._http.delete(f"/collections/{name}"))

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def ingest(self, collection: str, documents: list[dict[str, Any]]) -> IngestResponse:
        """Ingest documents synchronously (one by one via the sync endpoint)."""
        total_indexed = 0
        total_errors = 0
        for doc in documents:
            payload = {"document": doc}
            data = self._check(self._http.post(f"/collections/{collection}/documents", json=payload))
            total_indexed += data.get("indexed", 0)
            total_errors += data.get("errors", 0)
        return IngestResponse(indexed=total_indexed, errors=total_errors)

    def ingest_one(self, collection: str, document: dict[str, Any]) -> IngestResponse:
        data = self._check(
            self._http.post(f"/collections/{collection}/documents", json={"document": document})
        )
        return IngestResponse(**data)

    def bulk_ingest(self, collection: str, documents: list[dict[str, Any]]) -> BulkIngestResponse:
        """Submit a bulk async ingestion job. Requires Redis on the server."""
        data = self._check(
            self._http.post(f"/collections/{collection}/documents/bulk", json={"documents": documents})
        )
        return BulkIngestResponse(**data)

    def delete_document(self, collection: str, doc_id: str) -> None:
        self._check(self._http.delete(f"/collections/{collection}/documents/{doc_id}"))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        payload: dict[str, Any] = {"query": query, "top_k": top_k, "alpha": alpha}
        if filters:
            payload["filters"] = filters
        data = self._check(
            self._http.post(f"/collections/{collection}/search", json=payload)
        )
        return SearchResponse(**data)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> JobResponse:
        data = self._check(self._http.get(f"/jobs/{job_id}"))
        return JobResponse(**data)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Semantix":
        return self

    def __exit__(self, *args) -> None:
        self.close()
