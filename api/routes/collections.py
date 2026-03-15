"""Collection CRUD endpoints."""
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_store
from api.models import (
    CreateCollectionRequest, CollectionStats, CollectionListResponse
)
from core.collection import CollectionConfig, CollectionStore, IndexType

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionStats, status_code=201)
def create_collection(
    req: CreateCollectionRequest,
    store: CollectionStore = Depends(get_store),
):
    config = CollectionConfig(
        name=req.name,
        embedding_field=req.embedding_field,
        embedding_dim=req.embedding_dim,
        index_type=IndexType(req.index_type),
        provider=req.provider,
        hnsw_m=req.hnsw_m,
        hnsw_ef_construction=req.hnsw_ef_construction,
        hnsw_ef_search=req.hnsw_ef_search,
    )
    try:
        col = store.create(config)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return CollectionStats(**col.stats())


@router.get("", response_model=CollectionListResponse)
def list_collections(store: CollectionStore = Depends(get_store)):
    names = store.list()
    stats = []
    for name in names:
        col = store.get(name)
        if col:
            stats.append(CollectionStats(**col.stats()))
    return CollectionListResponse(collections=stats, total=len(stats))


@router.get("/{name}", response_model=CollectionStats)
def get_collection(name: str, store: CollectionStore = Depends(get_store)):
    col = store.get(name)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return CollectionStats(**col.stats())


@router.delete("/{name}", status_code=204)
def delete_collection(name: str, store: CollectionStore = Depends(get_store)):
    deleted = store.delete(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
