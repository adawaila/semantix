"""Collection and CollectionStore tests — 25+ tests."""
import os
import tempfile

import numpy as np
import pytest

from core.collection import (
    Collection, CollectionConfig, CollectionStore, IndexType, SearchResult
)


DIM = 32


def make_config(name: str = "test", index_type: IndexType = IndexType.BRUTE_FORCE) -> CollectionConfig:
    return CollectionConfig(
        name=name,
        embedding_field="description",
        embedding_dim=DIM,
        index_type=index_type,
        provider="local",
    )


def make_doc(id: str, desc: str, **extra) -> dict:
    return {"id": id, "description": desc, **extra}


def rand_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Collection — basic ops
# ---------------------------------------------------------------------------

class TestCollectionBasic:
    def test_add_document(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "wireless headphones"))
        assert col.get("d1") is not None

    def test_add_requires_id(self):
        col = Collection(make_config())
        with pytest.raises(ValueError):
            col.add({"description": "no id here"})

    def test_get_missing_returns_none(self):
        col = Collection(make_config())
        assert col.get("ghost") is None

    def test_add_with_embedding(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "bluetooth speaker"), embedding=rand_emb(0))
        s = col.stats()
        assert s["vector_count"] == 1

    def test_add_wrong_embedding_dim_raises(self):
        col = Collection(make_config())
        with pytest.raises(ValueError):
            col.add(make_doc("d1", "test"), embedding=np.ones(DIM + 1, dtype=np.float32))

    def test_add_overwrites(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "old text"))
        col.add(make_doc("d1", "new text"))
        assert col.stats()["doc_count"] == 1
        doc = col.get("d1")
        assert doc["description"] == "new text"

    def test_delete_existing(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "to delete"))
        deleted = col.delete("d1")
        assert deleted is True
        assert col.get("d1") is None

    def test_delete_nonexistent(self):
        col = Collection(make_config())
        assert col.delete("ghost") is False

    def test_add_batch(self):
        col = Collection(make_config())
        docs = [make_doc(str(i), f"document {i}") for i in range(10)]
        embs = np.stack([rand_emb(i) for i in range(10)])
        col.add_batch(docs, embeddings=embs)
        assert col.stats()["doc_count"] == 10
        assert col.stats()["vector_count"] == 10

    def test_add_batch_no_embeddings(self):
        col = Collection(make_config())
        docs = [make_doc(str(i), f"content {i}") for i in range(5)]
        col.add_batch(docs)
        assert col.stats()["doc_count"] == 5
        assert col.stats()["vector_count"] == 0

    def test_add_batch_length_mismatch(self):
        col = Collection(make_config())
        docs = [make_doc("d1", "x")]
        embs = np.stack([rand_emb(0), rand_emb(1)])
        with pytest.raises(ValueError):
            col.add_batch(docs, embeddings=embs)

    def test_stats(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "hello world"), embedding=rand_emb(0))
        s = col.stats()
        assert s["doc_count"] == 1
        assert s["vector_count"] == 1
        assert s["embedding_dim"] == DIM
        assert s["name"] == "test"


# ---------------------------------------------------------------------------
# Collection — search
# ---------------------------------------------------------------------------

class TestCollectionSearch:
    def test_bm25_only_search(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "wireless headphones noise cancelling"))
        col.add(make_doc("d2", "kitchen blender food processor"))
        results = col.search("wireless headphones", alpha=0.0)
        assert len(results) >= 1
        assert results[0].id == "d1"

    def test_vector_only_search(self):
        col = Collection(make_config())
        target_emb = rand_emb(99)
        col.add(make_doc("d1", "text"), embedding=target_emb)
        for i in range(5):
            col.add(make_doc(str(i), f"other {i}"), embedding=rand_emb(i))
        results = col.search("text", query_embedding=target_emb, alpha=1.0, top_k=1)
        assert results[0].id == "d1"

    def test_hybrid_search(self):
        col = Collection(make_config())
        embs = [rand_emb(i) for i in range(10)]
        for i in range(10):
            col.add(make_doc(str(i), f"product description number {i}"), embedding=embs[i])
        q_emb = rand_emb(0)
        results = col.search("product description", query_embedding=q_emb, alpha=0.5, top_k=5)
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_no_embedding_forces_bm25(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "special unique keyword term"))
        results = col.search("special unique keyword", alpha=0.5)
        assert any(r.id == "d1" for r in results)

    def test_top_k(self):
        col = Collection(make_config())
        for i in range(20):
            col.add(make_doc(str(i), f"search content item {i}"))
        results = col.search("search content", top_k=5)
        assert len(results) <= 5

    def test_search_empty_collection(self):
        col = Collection(make_config())
        results = col.search("anything")
        assert results == []

    def test_result_has_document(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "hello", price=9.99))
        results = col.search("hello")
        assert results[0].document["price"] == 9.99

    def test_query_count_increments(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "hello world"))
        col.search("hello")
        col.search("world")
        assert col.stats()["query_count"] == 2


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestCollectionFilters:
    def test_exact_match_filter(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "shoes", category="footwear"))
        col.add(make_doc("d2", "shirt", category="clothing"))
        results = col.search("shoes shirt", filters={"category": "footwear"})
        assert all(r.document["category"] == "footwear" for r in results)

    def test_range_filter_gte(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "cheap item", price=5.0))
        col.add(make_doc("d2", "expensive item", price=100.0))
        col.add(make_doc("d3", "mid item", price=50.0))
        results = col.search("item", filters={"price": {"$gte": 50.0}})
        prices = [r.document["price"] for r in results]
        assert all(p >= 50.0 for p in prices)

    def test_range_filter_lte(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "cheap item", price=5.0))
        col.add(make_doc("d2", "expensive item", price=100.0))
        results = col.search("item", filters={"price": {"$lte": 10.0}})
        assert all(r.document["price"] <= 10.0 for r in results)

    def test_contains_filter(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "sale item", tags=["sale", "electronics"]))
        col.add(make_doc("d2", "regular item", tags=["electronics"]))
        results = col.search("item", filters={"tags": {"$contains": "sale"}})
        assert all("sale" in r.document["tags"] for r in results)

    def test_ne_filter(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "item", status="active"))
        col.add(make_doc("d2", "item", status="inactive"))
        results = col.search("item", filters={"status": {"$ne": "inactive"}})
        assert all(r.document["status"] != "inactive" for r in results)

    def test_no_match_filter(self):
        col = Collection(make_config())
        col.add(make_doc("d1", "item", category="food"))
        results = col.search("item", filters={"category": "electronics"})
        assert results == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestCollectionPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config()
            col = Collection(config, data_dir=tmpdir)
            col.add(make_doc("d1", "hello world"), embedding=rand_emb(0))
            col.add(make_doc("d2", "foo bar baz"), embedding=rand_emb(1))
            col.save()

            path = os.path.join(tmpdir, "test.pkl")
            col2 = Collection.load(path)
            assert col2.stats()["doc_count"] == 2
            assert col2.get("d1") is not None
            assert col2.get("d2") is not None

    def test_search_after_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config()
            col = Collection(config, data_dir=tmpdir)
            col.add(make_doc("d1", "unique keyword frobble"))
            col.save()

            path = os.path.join(tmpdir, "test.pkl")
            col2 = Collection.load(path)
            results = col2.search("unique keyword frobble")
            assert any(r.id == "d1" for r in results)


# ---------------------------------------------------------------------------
# CollectionStore
# ---------------------------------------------------------------------------

class TestCollectionStore:
    def test_create_and_get(self):
        store = CollectionStore()
        store.create(make_config("products"))
        col = store.get("products")
        assert col is not None

    def test_create_duplicate_raises(self):
        store = CollectionStore()
        store.create(make_config("products"))
        with pytest.raises(ValueError):
            store.create(make_config("products"))

    def test_list(self):
        store = CollectionStore()
        store.create(make_config("a"))
        store.create(make_config("b"))
        assert set(store.list()) == {"a", "b"}

    def test_delete(self):
        store = CollectionStore()
        store.create(make_config("x"))
        deleted = store.delete("x")
        assert deleted is True
        assert store.get("x") is None

    def test_delete_nonexistent(self):
        store = CollectionStore()
        assert store.delete("ghost") is False

    def test_contains(self):
        store = CollectionStore()
        store.create(make_config("y"))
        assert "y" in store
        assert "z" not in store

    def test_persist_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CollectionStore(data_dir=tmpdir)
            store.create(make_config("products"))
            col = store.get("products")
            col.add(make_doc("d1", "test content"))
            col.save()

            # New store loads from disk
            store2 = CollectionStore(data_dir=tmpdir)
            assert "products" in store2
            col2 = store2.get("products")
            assert col2.get("d1") is not None
