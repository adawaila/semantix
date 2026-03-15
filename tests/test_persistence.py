"""Persistence tests — atomic writes, crash recovery, state restoration."""
from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest

from core.collection import Collection, CollectionConfig, CollectionStore, IndexType

DIM = 16


def rand_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_config(name: str = "test") -> CollectionConfig:
    return CollectionConfig(
        name=name,
        embedding_field="text",
        embedding_dim=DIM,
        index_type=IndexType.BRUTE_FORCE,
    )


def make_doc(id: str, text: str, **kw) -> dict:
    return {"id": id, "text": text, **kw}


# ---------------------------------------------------------------------------
# Atomic write tests
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_no_tmp_file_after_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Collection(make_config(), data_dir=tmpdir)
            col.add(make_doc("d1", "hello"))
            col.save()
            tmp_path = os.path.join(tmpdir, "test.pkl.tmp")
            assert not os.path.exists(tmp_path)

    def test_pkl_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Collection(make_config(), data_dir=tmpdir)
            col.save()
            assert os.path.exists(os.path.join(tmpdir, "test.pkl"))

    def test_in_memory_save_noop(self):
        col = Collection(make_config(), data_dir=None)
        col.add(make_doc("d1", "hello"))
        col.save()  # should not raise
        assert True

    def test_overwrite_is_atomic(self):
        """Saving twice should produce a clean file, no corruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Collection(make_config(), data_dir=tmpdir)
            col.add(make_doc("d1", "first"))
            col.save()
            col.add(make_doc("d2", "second"))
            col.save()
            col2 = Collection.load(os.path.join(tmpdir, "test.pkl"))
            assert col2.stats()["doc_count"] == 2


# ---------------------------------------------------------------------------
# Load/restore tests
# ---------------------------------------------------------------------------

class TestLoadRestore:
    def test_restore_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Collection(make_config(), data_dir=tmpdir)
            col.add(make_doc("d1", "important content", tag="alpha"))
            col.save()
            col2 = Collection.load(os.path.join(tmpdir, "test.pkl"))
            doc = col2.get("d1")
            assert doc is not None
            assert doc["tag"] == "alpha"

    def test_restore_bm25_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Collection(make_config(), data_dir=tmpdir)
            col.add(make_doc("d1", "unique frobble wizzle zorch"))
            col.save()
            col2 = Collection.load(os.path.join(tmpdir, "test.pkl"))
            results = col2.search("unique frobble wizzle", alpha=0.0)
            assert any(r.id == "d1" for r in results)

    def test_restore_vector_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Collection(make_config(), data_dir=tmpdir)
            target_emb = rand_emb(42)
            col.add(make_doc("d1", "target"), embedding=target_emb)
            col.add(make_doc("d2", "other"), embedding=rand_emb(1))
            col.save()
            col2 = Collection.load(os.path.join(tmpdir, "test.pkl"))
            results = col2.search("target", query_embedding=target_emb, alpha=1.0, top_k=1)
            assert results[0].id == "d1"

    def test_restore_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Collection(make_config(), data_dir=tmpdir)
            for i in range(5):
                col.add(make_doc(str(i), f"content {i}"), embedding=rand_emb(i))
            col.save()
            col2 = Collection.load(os.path.join(tmpdir, "test.pkl"))
            s = col2.stats()
            assert s["doc_count"] == 5
            assert s["vector_count"] == 5

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            Collection.load("/nonexistent/path/file.pkl")

    def test_restore_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config("myconfig")
            config.embedding_field = "body"
            col = Collection(config, data_dir=tmpdir)
            col.save()
            col2 = Collection.load(os.path.join(tmpdir, "myconfig.pkl"))
            assert col2.config.embedding_field == "body"
            assert col2.config.name == "myconfig"


# ---------------------------------------------------------------------------
# CollectionStore persistence
# ---------------------------------------------------------------------------

class TestStorePersistence:
    def test_store_auto_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = CollectionStore(data_dir=tmpdir)
            config = make_config("persistent_col")
            col = store1.create(config)
            col.add(make_doc("d1", "hello world"))
            col.save()

            store2 = CollectionStore(data_dir=tmpdir)
            assert "persistent_col" in store2
            doc = store2.get("persistent_col").get("d1")
            assert doc is not None

    def test_store_delete_removes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CollectionStore(data_dir=tmpdir)
            store.create(make_config("to_remove"))
            store.delete("to_remove")
            assert not os.path.exists(os.path.join(tmpdir, "to_remove.pkl"))

    def test_store_survives_corrupt_file(self):
        """CollectionStore skips corrupt .pkl files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt = os.path.join(tmpdir, "corrupt.pkl")
            with open(corrupt, "wb") as f:
                f.write(b"this is not valid pickle data")
            # Should not raise — just skip the corrupt file
            store = CollectionStore(data_dir=tmpdir)
            assert "corrupt" not in store

    def test_multiple_collections_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = CollectionStore(data_dir=tmpdir)
            for i in range(3):
                col = store1.create(make_config(f"col{i}"))
                col.add(make_doc("d1", f"content {i}"))
                col.save()

            store2 = CollectionStore(data_dir=tmpdir)
            assert len(store2.list()) == 3
