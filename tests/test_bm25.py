"""BM25 index test suite — 30+ tests covering scoring, edge cases, and threading."""
import math
import threading

import pytest

from core.bm25 import BM25Index, Tokenizer


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_basic_tokenization(self):
        tok = Tokenizer(use_stemming=False)
        tokens = tok.tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_stopword_removal(self):
        tok = Tokenizer(use_stemming=False)
        tokens = tok.tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_punctuation_stripped(self):
        tok = Tokenizer(use_stemming=False)
        tokens = tok.tokenize("hello, world! it's great.")
        assert "," not in " ".join(tokens)
        assert "!" not in " ".join(tokens)
        assert "hello" in tokens

    def test_lowercase(self):
        tok = Tokenizer(use_stemming=False)
        tokens = tok.tokenize("UPPER LOWER MiXeD")
        assert all(t == t.lower() for t in tokens)

    def test_empty_string(self):
        tok = Tokenizer(use_stemming=False)
        assert tok.tokenize("") == []

    def test_only_stopwords(self):
        tok = Tokenizer(use_stemming=False)
        assert tok.tokenize("the and or but") == []

    def test_stemming_enabled(self):
        tok = Tokenizer(use_stemming=True)
        tokens = tok.tokenize("running runners runs")
        # All forms should reduce to shorter common stem
        assert all(len(t) <= 7 for t in tokens)

    def test_unique_terms(self):
        tok = Tokenizer(use_stemming=False)
        terms = tok.unique_terms("hello hello world")
        assert isinstance(terms, set)
        assert "hello" in terms
        assert "world" in terms

    def test_short_word_filtered(self):
        tok = Tokenizer(use_stemming=False)
        tokens = tok.tokenize("I am OK")
        # Single-char tokens filtered; 'ok' is 2 chars so stays
        assert "i" not in tokens


# ---------------------------------------------------------------------------
# BM25Index — basic operations
# ---------------------------------------------------------------------------

class TestBM25Basic:
    def test_empty_index(self):
        idx = BM25Index()
        assert idx.doc_count() == 0
        assert idx.vocab_size() == 0
        assert idx.search("anything") == []

    def test_add_single(self):
        idx = BM25Index()
        idx.add("d1", "wireless headphones with noise cancelling")
        assert idx.doc_count() == 1
        assert idx.contains("d1")

    def test_add_multiple(self):
        idx = BM25Index()
        for i in range(5):
            idx.add(f"d{i}", f"document number {i} with unique content")
        assert idx.doc_count() == 5

    def test_ids(self):
        idx = BM25Index()
        idx.add("x", "hello world")
        idx.add("y", "foo bar")
        assert set(idx.ids()) == {"x", "y"}

    def test_contains_false(self):
        idx = BM25Index()
        assert idx.contains("ghost") is False

    def test_vocab_size(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "apple banana cherry")
        # At least these three terms indexed
        assert idx.vocab_size() >= 3

    def test_remove_existing(self):
        idx = BM25Index()
        idx.add("d1", "hello world")
        removed = idx.remove("d1")
        assert removed is True
        assert idx.doc_count() == 0
        assert not idx.contains("d1")

    def test_remove_nonexistent(self):
        idx = BM25Index()
        assert idx.remove("ghost") is False

    def test_add_overwrites(self):
        idx = BM25Index()
        idx.add("d1", "wireless headphones")
        idx.add("d1", "completely different content here")
        assert idx.doc_count() == 1
        results = idx.search("wireless")
        # Old content replaced — "wireless" should not score well (or score 0)
        assert results == [] or results[0][1] < 0.5

    def test_term_freq(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "apple apple apple banana")
        assert idx.term_freq("d1", "apple") == 3
        assert idx.term_freq("d1", "banana") == 1

    def test_term_freq_missing_doc(self):
        idx = BM25Index()
        assert idx.term_freq("ghost", "apple") == 0


# ---------------------------------------------------------------------------
# BM25Index — scoring
# ---------------------------------------------------------------------------

class TestBM25Scoring:
    def test_relevant_doc_scores_higher(self):
        idx = BM25Index(use_stemming=False)
        idx.add("relevant", "wireless headphones noise cancelling audio")
        idx.add("irrelevant", "kitchen utensils cooking pots pans")
        results = idx.search("wireless headphones")
        assert results[0][0] == "relevant"

    def test_more_matches_scores_higher(self):
        idx = BM25Index(use_stemming=False)
        idx.add("many", "python python python programming language")
        idx.add("few", "python syntax rules")
        results = idx.search("python")
        # "many" should score higher due to higher TF
        assert results[0][0] == "many"

    def test_scores_descending(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "machine learning deep neural networks")
        idx.add("d2", "machine learning algorithms")
        idx.add("d3", "cooking recipes food")
        results = idx.search("machine learning")
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self):
        idx = BM25Index()
        for i in range(20):
            idx.add(f"d{i}", f"search engine document {i} keyword content")
        results = idx.search("search engine", top_k=5)
        assert len(results) <= 5

    def test_top_k_larger_than_hits(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "apple")
        idx.add("d2", "banana")
        results = idx.search("apple", top_k=10)
        assert len(results) == 1  # only d1 matches

    def test_query_no_match(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "apple banana cherry")
        results = idx.search("zyxwvuts")
        assert results == []

    def test_empty_query(self):
        idx = BM25Index()
        idx.add("d1", "some content")
        assert idx.search("") == []

    def test_all_stopwords_query(self):
        idx = BM25Index()
        idx.add("d1", "some content")
        assert idx.search("the and or") == []

    def test_score_positive(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "hello world foo bar")
        results = idx.search("hello")
        assert len(results) == 1
        assert results[0][1] > 0

    def test_idf_rare_term_higher(self):
        """A term appearing in fewer docs should have higher IDF."""
        idx = BM25Index(use_stemming=False)
        # "rare" only in d1; "common" in all docs
        idx.add("d1", "rare common content")
        idx.add("d2", "common different words")
        idx.add("d3", "common another text")
        results_rare = idx.search("rare")
        results_common = idx.search("common")
        # rare term in a single doc → higher IDF → higher score for that doc
        assert results_rare[0][1] > results_common[0][1]

    def test_length_normalisation(self):
        """Shorter docs with the same TF should score higher (BM25 length norm)."""
        idx = BM25Index(use_stemming=False)
        idx.add("short", "python language")
        # Many filler words — same TF for "python" but much longer doc
        idx.add("long", "python " + " ".join([f"word{i}" for i in range(50)]))
        results = idx.search("python")
        assert results[0][0] == "short"


# ---------------------------------------------------------------------------
# BM25Index — remove and re-index
# ---------------------------------------------------------------------------

class TestBM25Remove:
    def test_remove_clears_postings(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "apple")
        idx.remove("d1")
        assert idx.search("apple") == []
        assert idx.vocab_size() == 0

    def test_re_add_after_remove(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "apple")
        idx.remove("d1")
        idx.add("d1", "banana")
        results = idx.search("banana")
        assert results[0][0] == "d1"

    def test_partial_remove(self):
        idx = BM25Index(use_stemming=False)
        idx.add("d1", "apple banana")
        idx.add("d2", "apple cherry")
        idx.remove("d1")
        results = idx.search("apple")
        assert len(results) == 1
        assert results[0][0] == "d2"


# ---------------------------------------------------------------------------
# Threading safety
# ---------------------------------------------------------------------------

class TestBM25Threading:
    def test_concurrent_adds(self):
        idx = BM25Index()
        errors: list[Exception] = []

        def add_docs(start: int) -> None:
            try:
                for i in range(start, start + 50):
                    idx.add(f"d{i}", f"document {i} content search engine")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_docs, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert idx.doc_count() == 200

    def test_concurrent_reads(self):
        idx = BM25Index()
        for i in range(50):
            idx.add(f"d{i}", f"query test document content {i}")

        results_list: list[list] = []
        errors: list[Exception] = []

        def search(_):
            try:
                r = idx.search("query test")
                results_list.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=search, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results_list) == 10
