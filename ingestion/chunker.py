"""Document chunker — splits large text fields into overlapping chunks.

Each chunk becomes an independent indexable unit. The chunk inherits all
fields from the parent document, with the embedding_field replaced by the
chunk text. A synthetic id is generated: "{doc_id}__chunk_{n}".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    """A single chunk derived from a parent document."""
    id: str             # "{parent_id}__chunk_{n}"
    parent_id: str      # original document id
    text: str           # chunk text (embedding_field content)
    chunk_index: int    # 0-based chunk index within the parent document
    document: dict[str, Any]   # full document dict with embedding_field replaced


class Chunker:
    """Splits a document's embedding_field into overlapping text chunks.

    Parameters
    ----------
    chunk_size:
        Approximate number of characters per chunk (default 512).
    chunk_overlap:
        Number of characters to overlap between consecutive chunks (default 64).
    embedding_field:
        The field to chunk. The full document (with this field replaced by
        the chunk text) is stored in each Chunk.
    min_chunk_size:
        Chunks shorter than this are dropped (default 20).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_field: str = "description",
        min_chunk_size: int = 20,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_field = embedding_field
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str) -> list[str]:
        """Split *text* into a list of overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if len(chunk.strip()) >= self.min_chunk_size:
                chunks.append(chunk)
            start += step

        return chunks or [text]

    def chunk_document(self, document: dict[str, Any]) -> list[Chunk]:
        """Produce a list of Chunk objects from a document.

        If the text fits in a single chunk, one Chunk is returned with
        chunk_index=0 and id equal to the parent id (no suffix).
        """
        doc_id = str(document.get("id", ""))
        text = str(document.get(self.embedding_field, ""))

        parts = self.chunk_text(text)
        if not parts:
            parts = [text or ""]

        chunks: list[Chunk] = []
        for i, part in enumerate(parts):
            if len(parts) == 1:
                chunk_id = doc_id  # no suffix for single-chunk documents
            else:
                chunk_id = f"{doc_id}__chunk_{i}"

            chunk_doc = dict(document)
            chunk_doc["id"] = chunk_id
            chunk_doc[self.embedding_field] = part

            chunks.append(Chunk(
                id=chunk_id,
                parent_id=doc_id,
                text=part,
                chunk_index=i,
                document=chunk_doc,
            ))

        return chunks
