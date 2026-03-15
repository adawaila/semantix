"""Ingestion worker process.

Each worker is an independent Python process that:
1. Pulls jobs from Redis (BRPOP on QUEUE_KEY)
2. Runs the IngestionPipeline
3. Updates job progress in Redis
4. Handles SIGTERM gracefully (finishes current batch, then exits)

Usage:
    python -m ingestion.worker --redis redis://localhost:6379 --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Any

import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [worker] %(message)s",
)
logger = logging.getLogger(__name__)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    logger.info("SIGTERM received — finishing current batch then exiting")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def run_worker(
    redis_url: str,
    data_dir: str,
    embed_provider: str = "local",
    batch_size: int = 32,
    worker_id: int = 0,
) -> None:
    """Main worker loop."""
    from core.collection import CollectionStore
    from embeddings import get_provider
    from ingestion.jobs import JobTracker, dequeue, JobStatus
    from ingestion.pipeline import IngestionPipeline

    r = redis.from_url(redis_url)
    store = CollectionStore(data_dir=data_dir)
    provider = get_provider(embed_provider)
    tracker = JobTracker(r)

    logger.info("Worker %d started (provider=%s)", worker_id, embed_provider)

    while not _shutdown:
        payload = dequeue(r, timeout=5)
        if payload is None:
            continue

        job_id = payload.get("job_id", "")
        collection_name = payload.get("collection", "")
        documents = payload.get("documents", [])

        logger.info(
            "Worker %d processing job %s: %d docs → collection '%s'",
            worker_id, job_id, len(documents), collection_name,
        )

        tracker.set_running(job_id)

        col = store.get(collection_name)
        if col is None:
            tracker.set_error(job_id, f"Collection '{collection_name}' not found")
            continue

        pipeline = IngestionPipeline(col, provider, embed_batch_size=batch_size)
        try:
            result = pipeline.ingest(documents)
            tracker.increment_done(job_id, result["indexed"])
            if result["errors"]:
                tracker.increment_errors(job_id, count=result["errors"])
            col.save()
            tracker.set_done(job_id)
            logger.info("Job %s done: %d indexed, %d errors", job_id, result["indexed"], result["errors"])
        except Exception as e:
            logger.exception("Job %s failed: %s", job_id, e)
            tracker.set_error(job_id, str(e))

    logger.info("Worker %d exiting cleanly", worker_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="semantix ingestion worker")
    parser.add_argument("--redis", default=os.environ.get("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    parser.add_argument("--provider", default=os.environ.get("EMBED_PROVIDER", "local"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.workers == 1:
        run_worker(
            redis_url=args.redis,
            data_dir=args.data_dir,
            embed_provider=args.provider,
            batch_size=args.batch_size,
            worker_id=0,
        )
    else:
        import multiprocessing
        procs = []
        for i in range(args.workers):
            p = multiprocessing.Process(
                target=run_worker,
                kwargs=dict(
                    redis_url=args.redis,
                    data_dir=args.data_dir,
                    embed_provider=args.provider,
                    batch_size=args.batch_size,
                    worker_id=i,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
