"""Convenience entrypoint: python main.py [--host HOST] [--port PORT] [--reload]"""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the semantix API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument(
        "--data-dir", default=os.environ.get("DATA_DIR"), help="Persistence directory"
    )
    parser.add_argument(
        "--embed-provider",
        default=os.environ.get("EMBED_PROVIDER", "local"),
        choices=["local", "openai"],
        help="Embedding provider (default: local)",
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_URL"),
        help="Redis URL for async ingestion (optional)",
    )
    args = parser.parse_args()

    # Forward CLI args to env so the module-level `app` picks them up
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir
    os.environ["EMBED_PROVIDER"] = args.embed_provider
    if args.redis_url:
        os.environ["REDIS_URL"] = args.redis_url

    print(f"semantix v0.1.0  ->  http://{args.host}:{args.port}")
    print(f"  provider : {args.embed_provider}")
    print(f"  data_dir : {args.data_dir or 'in-memory'}")
    print(f"  redis    : {args.redis_url or 'disabled'}")
    print()

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
