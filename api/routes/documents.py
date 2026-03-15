"""Document ingestion, listing, and deletion endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_store, get_provider_dep, get_tracker, get_redis, get_collection
from api.models import (
    IngestDocumentRequest, BulkIngestRequest, IngestResponse, BulkIngestResponse,
    DocumentListResponse,
)
from core.collection import Collection, CollectionStore
from embeddings.base import EmbeddingProvider
from ingestion.jobs import JobTracker, enqueue
from ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/collections", tags=["documents"])


@router.get("/{name}/documents", response_model=DocumentListResponse)
def list_documents(
    name: str,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    store: CollectionStore = Depends(get_store),
):
    col = store.get(name)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    docs, total = col.list_docs(limit=limit, offset=offset)
    return DocumentListResponse(documents=docs, total=total, limit=limit, offset=offset)


@router.post("/{name}/documents", response_model=IngestResponse, status_code=201)
def ingest_document(
    name: str,
    req: IngestDocumentRequest,
    store: CollectionStore = Depends(get_store),
    provider: EmbeddingProvider = Depends(get_provider_dep),
):
    col = store.get(name)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    pipeline = IngestionPipeline(col, provider, chunk_size=0)
    result = pipeline.ingest([req.document])
    if col.stats().get("doc_count", 0) > 0 and hasattr(col, 'save') and col.data_dir:
        col.save()
    return IngestResponse(**result)


@router.post("/{name}/documents/bulk", response_model=BulkIngestResponse, status_code=202)
def bulk_ingest(
    name: str,
    req: BulkIngestRequest,
    store: CollectionStore = Depends(get_store),
    tracker: JobTracker | None = Depends(get_tracker),
    redis_client=Depends(get_redis),
):
    col = store.get(name)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    if tracker is None or redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="Redis not configured — bulk ingestion unavailable. Use /documents for sync ingestion.",
        )

    job_id = tracker.create(total=len(req.documents))
    enqueue(redis_client, name, req.documents, job_id)
    return BulkIngestResponse(job_id=job_id, total=len(req.documents))


@router.delete("/{name}/documents/{doc_id}", status_code=204)
def delete_document(
    name: str,
    doc_id: str,
    store: CollectionStore = Depends(get_store),
):
    col = store.get(name)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    deleted = col.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    if col.data_dir:
        col.save()
