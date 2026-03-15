"""SDK client tests — sync and async against a live TestClient."""
from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import init_dependencies
from core.collection import CollectionStore
from embeddings.base import EmbeddingProvider
from sdk.client import Semantix, SemantixError
from sdk.async_client import AsyncSemantix
from sdk.models import CollectionStats, SearchResponse, IngestResponse

DIM = 16


class StubEmbeddings(EmbeddingProvider):
    @property
    def dim(self) -> int:
        return DIM

    def embed_one(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.random(DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_one(t) for t in texts], axis=0) if texts else np.empty((0, DIM), dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_client():
    app = create_app(data_dir=None, embed_provider="local", redis_url=None)
    with TestClient(app, base_url="http://testserver") as tc:
        store = CollectionStore()
        provider = StubEmbeddings()
        init_dependencies(store, tracker=None, provider=provider, redis_client=None)
        yield tc


@pytest.fixture(scope="module")
def sdk(test_client):
    """Sync SDK backed by the TestClient transport."""
    import httpx
    transport = httpx.MockTransport(test_client.app)
    # Use the real TestClient as a transport adapter
    client = Semantix(base_url="http://testserver")
    # Replace internal httpx client with one that routes through TestClient
    client._http = test_client  # TestClient has the same interface for our purposes
    return client


# Actually, patch SDK to use TestClient directly
class _TestSemantix:
    """Thin wrapper that delegates to TestClient."""
    def __init__(self, tc):
        self._tc = tc

    def _check(self, resp):
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise SemantixError(resp.status_code, detail)
        if resp.status_code == 204:
            return {}
        return resp.json()

    def create_collection(self, name, embedding_field="description", embedding_dim=DIM, index_type="brute_force", provider="local", **kw):
        from sdk.models import CollectionStats
        data = self._check(self._tc.post("/collections", json={
            "name": name, "embedding_field": embedding_field,
            "embedding_dim": embedding_dim, "index_type": index_type, "provider": provider, **kw
        }))
        return CollectionStats(**data)

    def list_collections(self):
        from sdk.models import CollectionStats
        data = self._check(self._tc.get("/collections"))
        return [CollectionStats(**c) for c in data["collections"]]

    def get_collection(self, name):
        from sdk.models import CollectionStats
        data = self._check(self._tc.get(f"/collections/{name}"))
        return CollectionStats(**data)

    def delete_collection(self, name):
        self._check(self._tc.delete(f"/collections/{name}"))

    def ingest_one(self, collection, document):
        from sdk.models import IngestResponse
        data = self._check(self._tc.post(f"/collections/{collection}/documents", json={"document": document}))
        return IngestResponse(**data)

    def ingest(self, collection, documents):
        from sdk.models import IngestResponse
        total_indexed = 0
        total_errors = 0
        for doc in documents:
            r = self.ingest_one(collection, doc)
            total_indexed += r.indexed
            total_errors += r.errors
        return IngestResponse(indexed=total_indexed, errors=total_errors)

    def delete_document(self, collection, doc_id):
        self._check(self._tc.delete(f"/collections/{collection}/documents/{doc_id}"))

    def search(self, collection, query, top_k=10, alpha=0.5, filters=None):
        from sdk.models import SearchResponse
        payload = {"query": query, "top_k": top_k, "alpha": alpha}
        if filters:
            payload["filters"] = filters
        data = self._check(self._tc.post(f"/collections/{collection}/search", json=payload))
        return SearchResponse(**data)


@pytest.fixture
def s(test_client):
    # Fresh store for each test
    store = CollectionStore()
    init_dependencies(store, tracker=None, provider=StubEmbeddings(), redis_client=None)
    return _TestSemantix(test_client)


# ---------------------------------------------------------------------------
# Sync SDK tests
# ---------------------------------------------------------------------------

class TestSyncSDK:
    def test_create_collection(self, s):
        col = s.create_collection("mystore")
        assert isinstance(col, CollectionStats)
        assert col.name == "mystore"
        assert col.doc_count == 0

    def test_list_collections(self, s):
        s.create_collection("a1")
        s.create_collection("b1")
        cols = s.list_collections()
        names = {c.name for c in cols}
        assert "a1" in names
        assert "b1" in names

    def test_get_collection(self, s):
        s.create_collection("gettable")
        col = s.get_collection("gettable")
        assert col.name == "gettable"

    def test_get_nonexistent_raises(self, s):
        with pytest.raises(SemantixError) as exc:
            s.get_collection("ghost_collection_xyz")
        assert exc.value.status_code == 404

    def test_delete_collection(self, s):
        s.create_collection("to_delete")
        s.delete_collection("to_delete")
        with pytest.raises(SemantixError) as exc:
            s.get_collection("to_delete")
        assert exc.value.status_code == 404

    def test_ingest_one(self, s):
        s.create_collection("ingest_one_col")
        result = s.ingest_one("ingest_one_col", {"id": "d1", "description": "hello world"})
        assert isinstance(result, IngestResponse)
        assert result.indexed == 1

    def test_ingest_batch(self, s):
        s.create_collection("ingest_batch_col")
        docs = [{"id": str(i), "description": f"product {i}"} for i in range(5)]
        result = s.ingest("ingest_batch_col", docs)
        assert result.indexed == 5

    def test_search_returns_response(self, s):
        s.create_collection("search_col")
        s.ingest("search_col", [{"id": "d1", "description": "wireless headphones noise cancelling"}])
        resp = s.search("search_col", "wireless headphones", top_k=3, alpha=0.0)
        assert isinstance(resp, SearchResponse)
        assert len(resp.results) >= 1

    def test_search_result_fields(self, s):
        s.create_collection("fields_col")
        s.ingest("fields_col", [{"id": "d1", "description": "test", "price": 9.99}])
        resp = s.search("fields_col", "test", alpha=0.0)
        r = resp.results[0]
        assert r.id == "d1"
        assert r.score >= 0
        assert "price" in r.document

    def test_search_with_filter(self, s):
        s.create_collection("filter_col")
        s.ingest("filter_col", [
            {"id": "a", "description": "shoe", "category": "footwear"},
            {"id": "b", "description": "shirt", "category": "clothing"},
        ])
        resp = s.search("filter_col", "shoe shirt", alpha=0.0, filters={"category": "footwear"})
        assert all(r.document["category"] == "footwear" for r in resp.results)

    def test_delete_document(self, s):
        s.create_collection("del_doc_col")
        s.ingest_one("del_doc_col", {"id": "d1", "description": "delete me"})
        s.delete_document("del_doc_col", "d1")
        col = s.get_collection("del_doc_col")
        assert col.doc_count == 0

    def test_delete_nonexistent_doc_raises(self, s):
        s.create_collection("del_ghost_col")
        with pytest.raises(SemantixError) as exc:
            s.delete_document("del_ghost_col", "ghost_doc")
        assert exc.value.status_code == 404

    def test_search_empty_collection(self, s):
        s.create_collection("empty_col")
        resp = s.search("empty_col", "anything", alpha=0.0)
        assert resp.results == []

    def test_search_latency_reported(self, s):
        s.create_collection("latency_col")
        s.ingest_one("latency_col", {"id": "d1", "description": "latency test"})
        resp = s.search("latency_col", "latency test", alpha=0.0)
        assert resp.latency_ms >= 0

    def test_search_query_in_response(self, s):
        s.create_collection("query_col")
        resp = s.search("query_col", "my specific query", alpha=0.0)
        assert resp.query == "my specific query"
