"""Async semantix Python client."""
from __future__ import annotations

from typing import Any

import httpx

from .client import SemantixError
from .models import (
    CollectionStats, SearchResponse, IngestResponse,
    BulkIngestResponse, JobResponse,
)


class AsyncSemantix:
    """Async semantix client using httpx.AsyncClient.

    Example
    -------
    >>> async with AsyncSemantix("http://localhost:8000") as client:
    ...     await client.create_collection("products")
    ...     results = await client.search("products", "headphones")
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
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, headers=headers)

    async def _check(self, resp: httpx.Response) -> dict:
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

    async def create_collection(
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
        data = await self._check(await self._http.post("/collections", json=payload))
        return CollectionStats(**data)

    async def list_collections(self) -> list[CollectionStats]:
        data = await self._check(await self._http.get("/collections"))
        return [CollectionStats(**c) for c in data["collections"]]

    async def get_collection(self, name: str) -> CollectionStats:
        data = await self._check(await self._http.get(f"/collections/{name}"))
        return CollectionStats(**data)

    async def delete_collection(self, name: str) -> None:
        await self._check(await self._http.delete(f"/collections/{name}"))

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    async def ingest_one(self, collection: str, document: dict[str, Any]) -> IngestResponse:
        data = await self._check(
            await self._http.post(f"/collections/{collection}/documents", json={"document": document})
        )
        return IngestResponse(**data)

    async def ingest(self, collection: str, documents: list[dict[str, Any]]) -> IngestResponse:
        total_indexed = 0
        total_errors = 0
        for doc in documents:
            result = await self.ingest_one(collection, doc)
            total_indexed += result.indexed
            total_errors += result.errors
        return IngestResponse(indexed=total_indexed, errors=total_errors)

    async def bulk_ingest(self, collection: str, documents: list[dict[str, Any]]) -> BulkIngestResponse:
        data = await self._check(
            await self._http.post(
                f"/collections/{collection}/documents/bulk",
                json={"documents": documents},
            )
        )
        return BulkIngestResponse(**data)

    async def delete_document(self, collection: str, doc_id: str) -> None:
        await self._check(await self._http.delete(f"/collections/{collection}/documents/{doc_id}"))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
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
        data = await self._check(
            await self._http.post(f"/collections/{collection}/search", json=payload)
        )
        return SearchResponse(**data)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def get_job(self, job_id: str) -> JobResponse:
        data = await self._check(await self._http.get(f"/jobs/{job_id}"))
        return JobResponse(**data)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncSemantix":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
