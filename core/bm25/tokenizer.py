"""Tokenizer with stopword removal and Porter stemming.

Designed for BM25 indexing — produces lowercase, stemmed tokens
with common English stopwords removed.
"""
from __future__ import annotations

import re
import string


# Minimal English stopwords — sufficient for BM25 keyword pruning
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "this", "that", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "as", "if",
    "than", "then", "there", "when", "where", "who", "which", "what", "how",
    "all", "each", "any", "some", "such", "into", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "once", "here", "only", "own", "same",
    "too", "very", "just", "about", "also",
})

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _porter_stem(word: str) -> str:
    """Minimal Porter-like stemmer — handles the most common suffixes."""
    if len(word) <= 3:
        return word

    # Step 1a
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s") and len(word) > 4:
        word = word[:-1]

    # Step 1b — eed / ed / ing
    if word.endswith("eed") and len(word) > 4:
        word = word[:-1]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "aeioul":
            word = word[:-1]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "aeioul":
            word = word[:-1]

    # Step 2 — common suffixes
    _step2 = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("iser", "ise"),
        ("alism", "al"), ("ization", "ize"), ("isation", "ise"),
        ("ation", "ate"), ("ator", "ate"), ("alism", "al"),
        ("alistic", "al"), ("fulness", "ful"), ("ousness", "ous"),
        ("iveness", "ive"), ("ization", "ize"),
    ]
    for suffix, replacement in _step2:
        if word.endswith(suffix) and len(word) - len(suffix) > 1:
            word = word[: -len(suffix)] + replacement
            break

    # Step 3 — final cleanup
    for suffix in ("icate", "ative", "alize", "iciti", "ical", "ful", "ness"):
        if word.endswith(suffix) and len(word) - len(suffix) > 1:
            word = word[: -len(suffix)]
            break

    return word


class Tokenizer:
    """Converts raw text to a list of normalised, stemmed tokens.

    Pipeline: lowercase → strip punctuation → split → remove stopwords → stem
    """

    def __init__(self, use_stemming: bool = True) -> None:
        self.use_stemming = use_stemming

    def tokenize(self, text: str) -> list[str]:
        """Return a list of processed tokens from *text*."""
        text = text.lower()
        text = _PUNCT_RE.sub(" ", text)
        tokens = _SPACE_RE.split(text.strip())
        tokens = [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]
        if self.use_stemming:
            tokens = [_porter_stem(t) for t in tokens]
        return tokens

    def unique_terms(self, text: str) -> set[str]:
        """Return the set of unique tokens from *text*."""
        return set(self.tokenize(text))
