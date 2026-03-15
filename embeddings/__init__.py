"""Embedding providers for semantix."""
from .base import EmbeddingProvider
from .local import LocalEmbeddings
from .openai import OpenAIEmbeddings


def get_provider(name: str, **kwargs) -> EmbeddingProvider:
    """Factory: return an EmbeddingProvider by name."""
    if name == "local":
        return LocalEmbeddings(**kwargs)
    elif name == "openai":
        return OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {name!r}. Choose 'local' or 'openai'.")


__all__ = ["EmbeddingProvider", "LocalEmbeddings", "OpenAIEmbeddings", "get_provider"]
