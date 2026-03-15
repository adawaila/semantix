"""FastAPI application factory."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import init_dependencies
from api.middleware import APIKeyMiddleware
from api.routes import collections, documents, search, jobs, ws
from core.collection import CollectionStore
from embeddings import get_provider


def create_app(
    data_dir: str | None = None,
    embed_provider: str = "local",
    redis_url: str | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    data_dir:
        Directory for persistent collection storage. None = in-memory.
    embed_provider:
        "local" or "openai".
    redis_url:
        Redis connection URL. None disables async bulk ingestion.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        store = CollectionStore(data_dir=data_dir)
        provider = get_provider(embed_provider)

        tracker = None
        redis_client = None
        if redis_url:
            try:
                import redis as redis_lib
                redis_client = redis_lib.from_url(redis_url)
                redis_client.ping()
                from ingestion.jobs import JobTracker
                tracker = JobTracker(redis_client)
            except Exception:
                pass  # Redis optional — bulk ingestion disabled

        init_dependencies(store, tracker, provider, redis_client)
        yield
        # Shutdown: save all collections
        for name in store.list():
            col = store.get(name)
            if col and col.data_dir:
                col.save()

    app = FastAPI(
        title="semantix",
        description="Self-hostable hybrid search engine (BM25 + vector) powered by vectr",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_key = os.environ.get("SEMANTIX_API_KEY")
    if api_key:
        app.add_middleware(APIKeyMiddleware, api_key=api_key)

    app.include_router(collections.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(jobs.router)
    app.include_router(ws.router)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


# Default app instance (used by uvicorn)
app = create_app(
    data_dir=os.environ.get("DATA_DIR"),
    embed_provider=os.environ.get("EMBED_PROVIDER", "local"),
    redis_url=os.environ.get("REDIS_URL"),
)
