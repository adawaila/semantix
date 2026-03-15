"""API endpoint tests — all routes, error cases, and WebSocket."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import init_dependencies
from core.collection import CollectionStore
from embeddings.base import EmbeddingProvider

DIM = 16


# ---------------------------------------------------------------------------
# Stub embedding provider
# ---------------------------------------------------------------------------

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
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app = create_app(data_dir=None, embed_provider="local", redis_url=None)
    with TestClient(app) as c:
        # Override dependencies AFTER lifespan has run (which sets real LocalEmbeddings)
        store = CollectionStore()
        provider = StubEmbeddings()
        init_dependencies(store, tracker=None, provider=provider, redis_client=None)
        yield c


@pytest.fixture
def client_with_collection(client):
    resp = client.post("/collections", json={
        "name": "products",
        "embedding_field": "description",
        "embedding_dim": DIM,
        "index_type": "brute_force",
        "provider": "local",
    })
    assert resp.status_code == 201
    return client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class TestCollections:
    def test_create_collection(self, client):
        resp = client.post("/collections", json={
            "name": "myindex",
            "embedding_field": "text",
            "embedding_dim": DIM,
            "index_type": "brute_force",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "myindex"
        assert data["doc_count"] == 0

    def test_create_duplicate_returns_409(self, client):
        payload = {"name": "dup", "embedding_dim": DIM, "index_type": "brute_force"}
        client.post("/collections", json=payload)
        resp = client.post("/collections", json=payload)
        assert resp.status_code == 409

    def test_create_invalid_name_returns_422(self, client):
        resp = client.post("/collections", json={"name": "bad name!", "embedding_dim": DIM})
        assert resp.status_code == 422

    def test_list_collections(self, client_with_collection):
        resp = client_with_collection.get("/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any(c["name"] == "products" for c in data["collections"])

    def test_get_collection(self, client_with_collection):
        resp = client_with_collection.get("/collections/products")
        assert resp.status_code == 200
        assert resp.json()["name"] == "products"

    def test_get_nonexistent_collection(self, client):
        resp = client.get("/collections/ghost")
        assert resp.status_code == 404

    def test_delete_collection(self, client_with_collection):
        resp = client_with_collection.delete("/collections/products")
        assert resp.status_code == 204
        assert client_with_collection.get("/collections/products").status_code == 404

    def test_delete_nonexistent_collection(self, client):
        resp = client.delete("/collections/ghost")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class TestDocuments:
    def test_ingest_document(self, client_with_collection):
        resp = client_with_collection.post(
            "/collections/products/documents",
            json={"document": {"id": "p1", "description": "wireless headphones", "price": 49.99}},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["indexed"] == 1
        assert data["errors"] == 0

    def test_ingest_without_id_returns_422(self, client_with_collection):
        resp = client_with_collection.post(
            "/collections/products/documents",
            json={"document": {"description": "no id"}},
        )
        assert resp.status_code == 422

    def test_ingest_to_nonexistent_collection(self, client):
        resp = client.post(
            "/collections/ghost/documents",
            json={"document": {"id": "d1", "description": "test"}},
        )
        assert resp.status_code == 404

    def test_bulk_ingest_no_redis_returns_503(self, client_with_collection):
        resp = client_with_collection.post(
            "/collections/products/documents/bulk",
            json={"documents": [{"id": "d1", "description": "test"}]},
        )
        assert resp.status_code == 503

    def test_delete_document(self, client_with_collection):
        client_with_collection.post(
            "/collections/products/documents",
            json={"document": {"id": "d1", "description": "delete me"}},
        )
        resp = client_with_collection.delete("/collections/products/documents/d1")
        assert resp.status_code == 204

    def test_delete_nonexistent_document(self, client_with_collection):
        resp = client_with_collection.delete("/collections/products/documents/ghost")
        assert resp.status_code == 404

    def test_ingest_updates_doc_count(self, client_with_collection):
        for i in range(5):
            client_with_collection.post(
                "/collections/products/documents",
                json={"document": {"id": str(i), "description": f"product {i}"}},
            )
        resp = client_with_collection.get("/collections/products")
        assert resp.json()["doc_count"] == 5


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    def _ingest(self, client, n: int = 5):
        for i in range(n):
            client.post(
                "/collections/products/documents",
                json={"document": {"id": str(i), "description": f"product item {i} keyword search"}},
            )

    def test_search_returns_results(self, client_with_collection):
        self._ingest(client_with_collection)
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": "product keyword", "top_k": 3, "alpha": 0.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_search_empty_collection(self, client_with_collection):
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": "anything", "top_k": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_search_top_k_respected(self, client_with_collection):
        self._ingest(client_with_collection, 10)
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": "product", "top_k": 3, "alpha": 0.0},
        )
        assert len(resp.json()["results"]) <= 3

    def test_search_nonexistent_collection(self, client):
        resp = client.post(
            "/collections/ghost/search",
            json={"query": "test"},
        )
        assert resp.status_code == 404

    def test_search_invalid_alpha(self, client_with_collection):
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": "test", "alpha": 1.5},
        )
        assert resp.status_code == 422

    def test_search_with_filter(self, client_with_collection):
        client_with_collection.post(
            "/collections/products/documents",
            json={"document": {"id": "a", "description": "shoes", "category": "footwear"}},
        )
        client_with_collection.post(
            "/collections/products/documents",
            json={"document": {"id": "b", "description": "shirt", "category": "clothing"}},
        )
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": "shoes shirt", "alpha": 0.0, "filters": {"category": "footwear"}},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert all(r["document"]["category"] == "footwear" for r in results)

    def test_search_result_structure(self, client_with_collection):
        self._ingest(client_with_collection, 3)
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": "product", "top_k": 1, "alpha": 0.0},
        )
        result = resp.json()["results"][0]
        assert "id" in result
        assert "score" in result
        assert "document" in result

    def test_search_empty_query_422(self, client_with_collection):
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": ""},
        )
        assert resp.status_code == 422

    def test_search_returns_total_docs(self, client_with_collection):
        self._ingest(client_with_collection, 7)
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": "product", "top_k": 3, "alpha": 0.0},
        )
        assert resp.json()["total_docs"] == 7

    def test_search_pure_bm25(self, client_with_collection):
        client_with_collection.post(
            "/collections/products/documents",
            json={"document": {"id": "x", "description": "unique frobble wizzle"}},
        )
        resp = client_with_collection.post(
            "/collections/products/search",
            json={"query": "unique frobble wizzle", "alpha": 0.0, "top_k": 1},
        )
        assert resp.json()["results"][0]["id"] == "x"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class TestJobs:
    def test_get_job_no_redis(self, client):
        resp = client.get("/jobs/some-job-id")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

class TestWebSocket:
    def test_ws_no_redis(self, client):
        with client.websocket_connect("/jobs/test-job/stream") as ws:
            data = ws.receive_text()
            msg = json.loads(data)
            assert "error" in msg


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

class TestListDocuments:
    def _ingest(self, c, n: int):
        for i in range(n):
            c.post(
                "/collections/products/documents",
                json={"document": {"id": str(i), "description": f"item {i}"}},
            )

    def test_list_documents_empty(self, client_with_collection):
        resp = client_with_collection.get("/collections/products/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"] == []
        assert data["total"] == 0

    def test_list_documents_returns_all(self, client_with_collection):
        self._ingest(client_with_collection, 5)
        resp = client_with_collection.get("/collections/products/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["documents"]) == 5

    def test_list_documents_pagination(self, client_with_collection):
        self._ingest(client_with_collection, 10)
        resp = client_with_collection.get("/collections/products/documents?limit=4&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["documents"]) == 4
        assert data["limit"] == 4
        assert data["offset"] == 0

    def test_list_documents_offset(self, client_with_collection):
        self._ingest(client_with_collection, 10)
        resp = client_with_collection.get("/collections/products/documents?limit=4&offset=8")
        data = resp.json()
        assert len(data["documents"]) == 2  # only 2 left

    def test_list_documents_not_found(self, client):
        resp = client.get("/collections/ghost/documents")
        assert resp.status_code == 404

    def test_embedding_field_in_stats(self, client_with_collection):
        resp = client_with_collection.get("/collections/products")
        assert resp.status_code == 200
        assert resp.json()["embedding_field"] == "description"


# ---------------------------------------------------------------------------
# API key middleware
# ---------------------------------------------------------------------------

class TestAPIKeyMiddleware:
    def test_no_key_set_allows_all(self, client):
        """When SEMANTIX_API_KEY is not set, all requests pass through."""
        resp = client.get("/collections")
        assert resp.status_code == 200

    def test_key_required_when_set(self, monkeypatch):
        monkeypatch.setenv("SEMANTIX_API_KEY", "secret123")
        app = create_app(data_dir=None, embed_provider="local", redis_url=None)
        with TestClient(app, raise_server_exceptions=False) as c:
            store = CollectionStore()
            init_dependencies(store, tracker=None, provider=StubEmbeddings(), redis_client=None)
            resp = c.get("/collections")
            assert resp.status_code == 401

    def test_bearer_token_accepted(self, monkeypatch):
        monkeypatch.setenv("SEMANTIX_API_KEY", "secret123")
        app = create_app(data_dir=None, embed_provider="local", redis_url=None)
        with TestClient(app) as c:
            store = CollectionStore()
            init_dependencies(store, tracker=None, provider=StubEmbeddings(), redis_client=None)
            resp = c.get("/collections", headers={"Authorization": "Bearer secret123"})
            assert resp.status_code == 200

    def test_x_api_key_header_accepted(self, monkeypatch):
        monkeypatch.setenv("SEMANTIX_API_KEY", "secret123")
        app = create_app(data_dir=None, embed_provider="local", redis_url=None)
        with TestClient(app) as c:
            store = CollectionStore()
            init_dependencies(store, tracker=None, provider=StubEmbeddings(), redis_client=None)
            resp = c.get("/collections", headers={"X-API-Key": "secret123"})
            assert resp.status_code == 200

    def test_wrong_key_rejected(self, monkeypatch):
        monkeypatch.setenv("SEMANTIX_API_KEY", "secret123")
        app = create_app(data_dir=None, embed_provider="local", redis_url=None)
        with TestClient(app, raise_server_exceptions=False) as c:
            store = CollectionStore()
            init_dependencies(store, tracker=None, provider=StubEmbeddings(), redis_client=None)
            resp = c.get("/collections", headers={"X-API-Key": "wrongkey"})
            assert resp.status_code == 401

    def test_health_exempt_from_auth(self, monkeypatch):
        monkeypatch.setenv("SEMANTIX_API_KEY", "secret123")
        app = create_app(data_dir=None, embed_provider="local", redis_url=None)
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
