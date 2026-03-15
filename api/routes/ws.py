"""WebSocket endpoint for live job progress streaming."""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.dependencies import get_tracker

router = APIRouter(tags=["websocket"])


@router.websocket("/jobs/{job_id}/stream")
async def job_stream(websocket: WebSocket, job_id: str):
    """Stream job progress updates until the job is done or errored."""
    await websocket.accept()
    tracker = get_tracker()

    if tracker is None:
        await websocket.send_text(json.dumps({"error": "Redis not configured"}))
        await websocket.close()
        return

    try:
        while True:
            info = tracker.get(job_id)
            if info is None:
                await websocket.send_text(json.dumps({"error": f"Job {job_id!r} not found"}))
                break

            payload = {
                "job_id": job_id,
                "status": info.status.value,
                "total": info.total,
                "done": info.done,
                "errors": info.errors,
                "progress": info.progress,
            }
            await websocket.send_text(json.dumps(payload))

            if info.status.value in ("done", "error"):
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
