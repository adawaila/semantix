"""Ingestion tests — chunker, pipeline, job tracker (Redis via testcontainers)."""
from __future__ import annotations

import numpy as np
import pytest

from core.collection import Collection, CollectionConfig, IndexType
from embeddings.base import EmbeddingProvider
from ingestion.chunker import Chunker, Chunk
from ingestion.pipeline import IngestionPipeline


DIM = 16  # small dim for speed


# ---------------------------------------------------------------------------
# Stub embedding provider — no model, no GPU
# ---------------------------------------------------------------------------

class StubEmbeddings(EmbeddingProvider):
    """Deterministic stub: hash(text) → unit vector."""

    @property
    def dim(self) -> int:
        return DIM

    def embed_one(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.random(DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_one(t) for t in texts], axis=0) if texts else np.empty((0, DIM), dtype=np.float32)


def make_config(name: str = "test") -> CollectionConfig:
    return CollectionConfig(
        name=name,
        embedding_field="description",
        embedding_dim=DIM,
        index_type=IndexType.BRUTE_FORCE,
    )


def make_doc(id: str, text: str, **kw) -> dict:
    return {"id": id, "description": text, **kw}


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_short_text_single_chunk(self):
        c = Chunker(chunk_size=512, embedding_field="description")
        chunks = c.chunk_document(make_doc("d1", "short text"))
        assert len(chunks) == 1
        assert chunks[0].id == "d1"  # no suffix for single chunk
        assert chunks[0].parent_id == "d1"

    def test_long_text_multiple_chunks(self):
        c = Chunker(chunk_size=50, chunk_overlap=10, embedding_field="description")
        text = "word " * 100  # 500 chars
        chunks = c.chunk_document(make_doc("d1", text))
        assert len(chunks) > 1

    def test_chunk_ids_suffixed(self):
        c = Chunker(chunk_size=50, chunk_overlap=10, embedding_field="description")
        text = "word " * 100
        chunks = c.chunk_document(make_doc("d1", text))
        for i, ch in enumerate(chunks):
            assert ch.id == f"d1__chunk_{i}"
            assert ch.chunk_index == i

    def test_chunk_text_replaced(self):
        c = Chunker(chunk_size=50, chunk_overlap=0, embedding_field="description")
        text = "A" * 60
        chunks = c.chunk_document(make_doc("d1", text))
        # Each chunk's doc should have description = chunk text
        for ch in chunks:
            assert ch.document["description"] == ch.text

    def test_chunk_preserves_other_fields(self):
        c = Chunker(chunk_size=50, chunk_overlap=10, embedding_field="description")
        doc = make_doc("d1", "short", price=9.99, category="tech")
        chunks = c.chunk_document(doc)
        assert chunks[0].document["price"] == 9.99
        assert chunks[0].document["category"] == "tech"

    def test_overlap_creates_continuity(self):
        c = Chunker(chunk_size=20, chunk_overlap=5, embedding_field="description")
        text = "A" * 100
        texts = c.chunk_text(text)
        assert len(texts) > 1
        # Adjacent chunks share overlap
        for i in range(len(texts) - 1):
            assert texts[i][-5:] == texts[i + 1][:5]

    def test_overlap_ge_size_raises(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=10)

    def test_empty_text(self):
        c = Chunker(chunk_size=512, embedding_field="description")
        chunks = c.chunk_document(make_doc("d1", ""))
        assert len(chunks) == 1
        assert chunks[0].text == ""

    def test_chunk_text_short(self):
        c = Chunker(chunk_size=512)
        texts = c.chunk_text("hello world")
        assert texts == ["hello world"]

    def test_chunk_text_empty(self):
        c = Chunker(chunk_size=512)
        assert c.chunk_text("") == []

    def test_min_chunk_size_filters(self):
        c = Chunker(chunk_size=30, chunk_overlap=0, min_chunk_size=20)
        # Last chunk might be too short
        text = "word " * 10  # 50 chars → 1 chunk of 30 + 1 of 20
        texts = c.chunk_text(text)
        # All returned chunks should be >= min_chunk_size OR the only chunk
        for t in texts:
            assert len(t.strip()) >= 20 or len(texts) == 1


# ---------------------------------------------------------------------------
# IngestionPipeline tests
# ---------------------------------------------------------------------------

class TestIngestionPipeline:
    def make_pipeline(self, chunk_size: int = 0, chunk_overlap: int = 64) -> tuple[IngestionPipeline, Collection]:
        col = Collection(make_config())
        provider = StubEmbeddings()
        pipeline = IngestionPipeline(col, provider, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embed_batch_size=8)
        return pipeline, col

    def test_ingest_single(self):
        pipeline, col = self.make_pipeline()
        result = pipeline.ingest([make_doc("d1", "hello world")])
        assert result["indexed"] == 1
        assert result["errors"] == 0
        assert col.get("d1") is not None

    def test_ingest_batch(self):
        pipeline, col = self.make_pipeline()
        docs = [make_doc(str(i), f"document {i} content") for i in range(20)]
        result = pipeline.ingest(docs)
        assert result["indexed"] == 20
        assert col.stats()["doc_count"] == 20

    def test_ingest_creates_embeddings(self):
        pipeline, col = self.make_pipeline()
        pipeline.ingest([make_doc("d1", "hello")])
        assert col.stats()["vector_count"] == 1

    def test_ingest_empty(self):
        pipeline, col = self.make_pipeline()
        result = pipeline.ingest([])
        assert result["indexed"] == 0

    def test_ingest_with_chunking(self):
        pipeline, col = self.make_pipeline(chunk_size=50, chunk_overlap=10)
        long_text = "word " * 100
        result = pipeline.ingest([make_doc("d1", long_text)])
        # Should have created multiple chunks
        assert result["indexed"] > 1

    def test_ingest_one(self):
        pipeline, col = self.make_pipeline()
        pipeline.ingest_one(make_doc("d1", "test"))
        assert col.get("d1") is not None

    def test_ingest_large_batch_batched_embedding(self):
        pipeline, col = self.make_pipeline()
        docs = [make_doc(str(i), f"content {i}") for i in range(50)]
        result = pipeline.ingest(docs)
        assert result["indexed"] == 50

    def test_search_after_ingest(self):
        pipeline, col = self.make_pipeline()
        pipeline.ingest([
            make_doc("d1", "wireless headphones noise cancelling"),
            make_doc("d2", "kitchen blender food processor"),
        ])
        results = col.search("headphones", alpha=0.0)
        assert any(r.id == "d1" for r in results)

    def test_missing_id_handled(self):
        pipeline, col = self.make_pipeline()
        # Document without "id" should count as error, not crash
        result = pipeline.ingest([{"description": "no id field"}])
        assert result["errors"] >= 1


# ---------------------------------------------------------------------------
# Job tracker — uses fakeredis (no Docker/server needed)
# ---------------------------------------------------------------------------

import fakeredis
from ingestion.jobs import JobTracker, JobStatus, enqueue, dequeue


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


class TestJobTracker:
    def test_create_and_get(self, fake_redis):
        tracker = JobTracker(fake_redis, ttl=60)
        job_id = tracker.create(total=10)
        info = tracker.get(job_id)
        assert info is not None
        assert info.status == JobStatus.QUEUED
        assert info.total == 10
        assert info.done == 0

    def test_increment_done(self, fake_redis):
        tracker = JobTracker(fake_redis, ttl=60)
        job_id = tracker.create(total=5)
        tracker.set_running(job_id)
        tracker.increment_done(job_id, 3)
        info = tracker.get(job_id)
        assert info.done == 3

    def test_auto_done_when_complete(self, fake_redis):
        tracker = JobTracker(fake_redis, ttl=60)
        job_id = tracker.create(total=3)
        tracker.set_running(job_id)
        tracker.increment_done(job_id, 3)
        info = tracker.get(job_id)
        assert info.status == JobStatus.DONE

    def test_set_error(self, fake_redis):
        tracker = JobTracker(fake_redis, ttl=60)
        job_id = tracker.create(total=5)
        tracker.set_error(job_id, "something broke")
        info = tracker.get(job_id)
        assert info.status == JobStatus.ERROR
        assert "broke" in info.error_msg

    def test_get_nonexistent(self, fake_redis):
        tracker = JobTracker(fake_redis, ttl=60)
        assert tracker.get("nonexistent-job-id-xyz") is None

    def test_enqueue_dequeue(self, fake_redis):
        tracker = JobTracker(fake_redis, ttl=60)
        job_id = tracker.create(total=2)
        docs = [{"id": "d1", "text": "hello"}, {"id": "d2", "text": "world"}]
        enqueue(fake_redis, "mycollection", docs, job_id)
        payload = dequeue(fake_redis, timeout=2)
        assert payload is not None
        assert payload["job_id"] == job_id
        assert payload["collection"] == "mycollection"
        assert len(payload["documents"]) == 2
