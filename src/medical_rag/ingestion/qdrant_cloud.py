"""Qdrant Cloud connection helpers.

Use this when running embeddings/indexing on Colab but storing vectors in a
persistent Qdrant Cloud free-tier cluster.

Environment variables:
    QDRANT_URL=https://xxxx.cloud.qdrant.io
    QDRANT_API_KEY=...
"""
from __future__ import annotations

import os
from typing import Any


def get_qdrant_cloud_client(
    url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> Any:
    """Create a Qdrant Cloud client.

    Args:
        url: Qdrant Cloud URL. Falls back to QDRANT_URL.
        api_key: Qdrant API key. Falls back to QDRANT_API_KEY.
        timeout: request timeout in seconds.

    Returns:
        qdrant_client.QdrantClient

    Raises:
        RuntimeError: if qdrant-client is missing or credentials are absent.
    """
    url = url or os.environ.get("QDRANT_URL")
    api_key = api_key or os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError(
            "Qdrant Cloud credentials missing. Set QDRANT_URL and QDRANT_API_KEY."
        )

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("Install Qdrant client: pip install qdrant-client") from exc

    client = QdrantClient(url=url, api_key=api_key, timeout=timeout, check_compatibility=False)
    client.get_collections()  # health check
    return client
