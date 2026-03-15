"""Job tracking for async ingestion — backed by Redis hash.

Job schema (stored as Redis hash fields):
    status:   "queued" | "running" | "done" | "error"
    total:    total documents submitted
    done:     documents successfully indexed
    errors:   number of errors
    error_msg: last error message (if any)
    created_at: Unix timestamp
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class JobInfo:
    job_id: str
    status: JobStatus
    total: int
    done: int
    errors: int
    error_msg: str
    created_at: float

    @property
    def progress(self) -> float:
        if self.total == 0:
            return 0.0
        return self.done / self.total


class JobTracker:
    """Tracks ingestion job progress in Redis.

    Parameters
    ----------
    redis_client:
        A redis.Redis instance.
    ttl:
        Key expiry in seconds (default 24 hours).
    """

    KEY_PREFIX = "semantix:job:"

    def __init__(self, redis_client: Any, ttl: int = 86400) -> None:
        self._redis = redis_client
        self._ttl = ttl

    def _key(self, job_id: str) -> str:
        return f"{self.KEY_PREFIX}{job_id}"

    def create(self, total: int) -> str:
        """Create a new job and return its job_id."""
        job_id = str(uuid.uuid4())
        key = self._key(job_id)
        self._redis.hset(key, mapping={
            "status": JobStatus.QUEUED.value,
            "total": str(total),
            "done": "0",
            "errors": "0",
            "error_msg": "",
            "created_at": str(time.time()),
        })
        self._redis.expire(key, self._ttl)
        return job_id

    def get(self, job_id: str) -> JobInfo | None:
        """Return job info, or None if not found."""
        key = self._key(job_id)
        data = self._redis.hgetall(key)
        if not data:
            return None
        # Redis returns bytes or strings depending on decode_responses setting
        def s(v) -> str:
            return v.decode() if isinstance(v, bytes) else str(v)

        # Handle both bytes keys and string keys
        d = {(k.decode() if isinstance(k, bytes) else k): s(v) for k, v in data.items()}
        return JobInfo(
            job_id=job_id,
            status=JobStatus(d.get("status", "queued")),
            total=int(d.get("total", 0)),
            done=int(d.get("done", 0)),
            errors=int(d.get("errors", 0)),
            error_msg=d.get("error_msg", ""),
            created_at=float(d.get("created_at", 0)),
        )

    def set_running(self, job_id: str) -> None:
        key = self._key(job_id)
        self._redis.hset(key, "status", JobStatus.RUNNING.value)

    def increment_done(self, job_id: str, count: int = 1) -> None:
        key = self._key(job_id)
        self._redis.hincrby(key, "done", count)
        # Check if complete
        data = self._redis.hmget(key, ["done", "total"])
        done = int(data[0] or 0)
        total = int(data[1] or 0)
        if total > 0 and done >= total:
            self._redis.hset(key, "status", JobStatus.DONE.value)

    def increment_errors(self, job_id: str, msg: str = "", count: int = 1) -> None:
        key = self._key(job_id)
        self._redis.hincrby(key, "errors", count)
        if msg:
            self._redis.hset(key, "error_msg", msg)

    def set_done(self, job_id: str) -> None:
        key = self._key(job_id)
        self._redis.hset(key, "status", JobStatus.DONE.value)

    def set_error(self, job_id: str, msg: str) -> None:
        key = self._key(job_id)
        self._redis.hset(key, mapping={"status": JobStatus.ERROR.value, "error_msg": msg})


QUEUE_KEY = "semantix:ingest_queue"


def enqueue(redis_client: Any, collection_name: str, documents: list[dict], job_id: str) -> None:
    """Push a batch job onto the Redis queue."""
    payload = json.dumps({
        "job_id": job_id,
        "collection": collection_name,
        "documents": documents,
    })
    redis_client.lpush(QUEUE_KEY, payload)


def dequeue(redis_client: Any, timeout: int = 5) -> dict | None:
    """Blocking pop from the Redis queue. Returns None on timeout."""
    result = redis_client.brpop(QUEUE_KEY, timeout=timeout)
    if result is None:
        return None
    _, payload = result
    if isinstance(payload, bytes):
        payload = payload.decode()
    return json.loads(payload)
