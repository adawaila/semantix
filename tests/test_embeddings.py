"""Embedding provider tests — LocalEmbeddings only (no OpenAI calls)."""
import numpy as np
import pytest

from embeddings import LocalEmbeddings, get_provider
from embeddings.base import EmbeddingProvider


# ---------------------------------------------------------------------------
# LocalEmbeddings
# ---------------------------------------------------------------------------

class TestLocalEmbeddings:
    @pytest.fixture(scope="class")
    def provider(self) -> LocalEmbeddings:
        return LocalEmbeddings()

    def test_is_embedding_provider(self, provider):
        assert isinstance(provider, EmbeddingProvider)

    def test_dim(self, provider):
        assert provider.dim == 384

    def test_embed_one_shape(self, provider):
        emb = provider.embed_one("hello world")
        assert emb.shape == (384,)

    def test_embed_one_dtype(self, provider):
        emb = provider.embed_one("test")
        assert emb.dtype == np.float32

    def test_embed_one_normalised(self, provider):
        emb = provider.embed_one("hello world")
        norm = np.linalg.norm(emb)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_embed_batch_shape(self, provider):
        texts = ["hello", "world", "foo"]
        embs = provider.embed_batch(texts)
        assert embs.shape == (3, 384)

    def test_embed_batch_dtype(self, provider):
        embs = provider.embed_batch(["a", "b"])
        assert embs.dtype == np.float32

    def test_embed_batch_normalised(self, provider):
        embs = provider.embed_batch(["hello", "world"])
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_batch_empty(self, provider):
        embs = provider.embed_batch([])
        assert embs.shape == (0, 384)

    def test_similar_texts_high_similarity(self, provider):
        a = provider.embed_one("wireless bluetooth headphones")
        b = provider.embed_one("bluetooth wireless headphones noise cancelling")
        sim = float(np.dot(a, b))
        assert sim > 0.7

    def test_dissimilar_texts_lower_similarity(self, provider):
        a = provider.embed_one("wireless bluetooth headphones")
        b = provider.embed_one("cooking recipe pasta tomato sauce")
        sim = float(np.dot(a, b))
        assert sim < 0.6

    def test_embed_batch_consistent_with_embed_one(self, provider):
        texts = ["hello world", "foo bar"]
        batch = provider.embed_batch(texts)
        for i, t in enumerate(texts):
            single = provider.embed_one(t)
            np.testing.assert_allclose(batch[i], single, atol=1e-5)

    def test_embed_one_long_text(self, provider):
        long_text = " ".join(["word"] * 200)
        emb = provider.embed_one(long_text)
        assert emb.shape == (384,)
        assert not np.any(np.isnan(emb))

    def test_embed_one_special_chars(self, provider):
        emb = provider.embed_one("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert emb.shape == (384,)

    def test_embed_one_empty_string(self, provider):
        # Should not raise; model handles empty input
        emb = provider.embed_one("")
        assert emb.shape == (384,)

    def test_lazy_model_loading(self):
        p = LocalEmbeddings()
        assert p._model is None  # not loaded yet
        _ = p.embed_one("trigger load")
        assert p._model is not None

    def test_thread_safety(self, provider):
        import threading
        results = []
        errors = []

        def embed():
            try:
                emb = provider.embed_one("concurrent embedding test")
                results.append(emb)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=embed) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 5
        for r in results:
            assert r.shape == (384,)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_local_provider(self):
        p = get_provider("local")
        assert isinstance(p, LocalEmbeddings)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            get_provider("unknown_xyz")
