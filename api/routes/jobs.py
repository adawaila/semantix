"""Job status endpoint."""
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_tracker
from api.models import JobResponse
from ingestion.jobs import JobTracker

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, tracker: JobTracker | None = Depends(get_tracker)):
    if tracker is None:
        raise HTTPException(status_code=503, detail="Redis not configured")
    info = tracker.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobResponse(
        job_id=info.job_id,
        status=info.status.value,
        total=info.total,
        done=info.done,
        errors=info.errors,
        error_msg=info.error_msg,
        progress=info.progress,
    )
