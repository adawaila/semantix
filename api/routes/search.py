"""Search endpoint."""
import time

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_store, get_provider_dep
from api.models import SearchRequest, SearchResponse, SearchResultItem
from core.collection import CollectionStore
from embeddings.base import EmbeddingProvider

router = APIRouter(prefix="/collections", tags=["search"])


@router.post("/{name}/search", response_model=SearchResponse)
def search(
    name: str,
    req: SearchRequest,
    store: CollectionStore = Depends(get_store),
    provider: EmbeddingProvider = Depends(get_provider_dep),
):
    col = store.get(name)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    t0 = time.perf_counter()

    # Embed query for vector search (only if alpha > 0)
    query_embedding = None
    if req.alpha > 0.0:
        try:
            query_embedding = provider.embed_one(req.query)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    results = col.search(
        query_text=req.query,
        query_embedding=query_embedding,
        top_k=req.top_k,
        alpha=req.alpha,
        filters=req.filters,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        results=[
            SearchResultItem(
                id=r.id,
                score=r.score,
                document=r.document,
                bm25_rank=r.bm25_rank,
                vector_rank=r.vector_rank,
            )
            for r in results
        ],
        total_docs=col.stats()["doc_count"],
        latency_ms=round(latency_ms, 2),
        query=req.query,
    )
