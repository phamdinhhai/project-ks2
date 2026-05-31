"""Qdrant multi-vector indexing for medical RAG.

Creates two Qdrant collections:
- text_chunks: 1024-dim BioMedBERT vectors + BM25 payload
- image_patches: 512-dim BioCLIP vectors (full + 4 quadrant patches)

Both collections keep the full payload for direct retrieval without
a secondary lookup, enabling fast evidence formatting in the agent.

Reference: Action plan specifies Qdrant for multi-vector support, which
enables simultaneous text and image retrieval with proper score normalization.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "image_patches"
TEXT_DIM = 1024   # BioMedBERT-large
IMAGE_DIM = 512   # BioCLIP


def _qdrant_client(url: str = "http://localhost:6333") -> Any:
    """Create a Qdrant client."""
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(url=url)
    except ImportError as exc:
        raise RuntimeError(
            "Qdrant client missing. Install: pip install qdrant-client"
        ) from exc


def _in_memory_client() -> Any:
    """Create an in-memory Qdrant client (no Docker required)."""
    from qdrant_client import QdrantClient
    return QdrantClient(":memory:")


def get_client(url: str | None = None) -> Any:
    """Get Qdrant client: try remote URL, fallback to in-memory."""
    if url and url != ":memory:":
        try:
            client = _qdrant_client(url)
            client.get_collections()  # health check
            logger.info(f"Connected to Qdrant at {url}")
            return client
        except Exception as exc:
            logger.warning(f"Qdrant at {url} not reachable ({exc}), using in-memory")
    logger.info("Using Qdrant in-memory (no persistence)")
    return _in_memory_client()


def init_qdrant_collections(client: Any, recreate: bool = False) -> None:
    """Create text_chunks and image_patches collections if they don't exist.

    Args:
        client: Qdrant client
        recreate: if True, drop and recreate existing collections
    """
    from qdrant_client.models import Distance, VectorParams

    existing = {c.name for c in client.get_collections().collections}

    for name, dim in [(TEXT_COLLECTION, TEXT_DIM), (IMAGE_COLLECTION, IMAGE_DIM)]:
        if name in existing:
            if recreate:
                client.delete_collection(name)
                logger.info(f"Dropped collection: {name}")
            else:
                logger.info(f"Collection already exists: {name}")
                continue
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {name} (dim={dim})")


def index_text_chunks(
    client: Any,
    documents: list[Any],  # list[DocumentChunk]
    encoder: Any,          # BioMedBERTEncoder
    chunk_size: int = 3,
    chunk_overlap: int = 1,
    batch_size: int = 32,
) -> int:
    """Index document chunks into text_chunks collection.

    Each DocumentChunk is split into sentence-level windows and each
    chunk gets its own vector point with full payload for retrieval.

    Args:
        client: Qdrant client
        documents: list of DocumentChunk objects
        encoder: BioMedBERTEncoder
        chunk_size: sentences per text chunk
        chunk_overlap: sentence overlap between chunks
        batch_size: encoding batch size

    Returns:
        total number of points indexed
    """
    from qdrant_client.models import PointStruct

    from medical_rag.ingestion.chunker import chunk_text

    all_texts: list[str] = []
    all_payloads: list[dict] = []

    for doc in documents:
        chunks = chunk_text(doc.text, window_size=chunk_size, overlap=chunk_overlap)
        for chunk_idx, chunk_text_str in enumerate(chunks):
            all_texts.append(chunk_text_str)
            all_payloads.append({
                "doc_id": doc.id,
                "chunk_idx": chunk_idx,
                "text": chunk_text_str,
                "dataset": doc.dataset,
                "source_path": doc.source_path,
                "record_id": doc.metadata.get("record_id", ""),
                "base_record_id": doc.metadata.get("base_record_id", ""),
                "question": doc.metadata.get("question", ""),
                "answer": doc.metadata.get("answer", ""),
            })

    total = 0
    for start in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[start : start + batch_size]
        batch_payloads = all_payloads[start : start + batch_size]
        vectors = encoder.encode_batch(batch_texts, normalize=True)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload=batch_payloads[i],
            )
            for i in range(len(batch_texts))
        ]
        client.upsert(collection_name=TEXT_COLLECTION, points=points)
        total += len(points)
        logger.info(f"Indexed text chunks: {total}/{len(all_texts)}")

    return total


def index_image_patches(
    client: Any,
    images: list[Any],   # list[ImageRecord]
    encoder: Any,        # BioCLIPEncoder
    batch_size: int = 16,
    use_patches: bool = True,
) -> int:
    """Index image patches into image_patches collection.

    Each ImageRecord produces up to 5 points (full + 4 quadrants)
    when use_patches=True, otherwise 1 point per image (caption text embedding).

    Args:
        client: Qdrant client
        images: list of ImageRecord objects
        encoder: BioCLIPEncoder
        batch_size: encoding batch size
        use_patches: if True, split image into 5 patches; else encode caption only

    Returns:
        total number of points indexed
    """
    from qdrant_client.models import PointStruct

    from medical_rag.ingestion.chunker import chunk_image

    all_vectors: list[list[float]] = []
    all_payloads: list[dict] = []

    for img_record in images:
        base_payload = {
            "image_id": img_record.id,
            "caption": img_record.caption,
            "dataset": img_record.dataset,
            "image_path": img_record.image_path,
            "record_id": img_record.metadata.get("record_id", ""),
            "base_record_id": img_record.metadata.get("base_record_id", ""),
        }

        if use_patches and img_record.image_path:
            path = Path(img_record.image_path)
            if path.exists():
                try:
                    patches = chunk_image(path)
                    for patch in patches:
                        vec = encoder.encode_image(patch["image"]).tolist()
                        payload = {
                            **base_payload,
                            "patch_id": patch["patch_id"],
                            "bbox": list(patch["bbox"]),
                        }
                        all_vectors.append(vec)
                        all_payloads.append(payload)
                    continue
                except Exception as exc:
                    logger.warning(f"Failed to load image {img_record.image_path}: {exc}")

        # Fallback: encode caption as text via BioCLIP text encoder
        vec = encoder.encode_text(img_record.caption or "medical image").tolist()
        payload = {**base_payload, "patch_id": "caption_fallback", "bbox": [0.0, 0.0, 1.0, 1.0]}
        all_vectors.append(vec)
        all_payloads.append(payload)

    total = 0
    for start in range(0, len(all_vectors), batch_size):
        batch_vecs = all_vectors[start : start + batch_size]
        batch_pays = all_payloads[start : start + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=batch_vecs[i],
                payload=batch_pays[i],
            )
            for i in range(len(batch_vecs))
        ]
        client.upsert(collection_name=IMAGE_COLLECTION, points=points)
        total += len(points)
        logger.info(f"Indexed image patches: {total}/{len(all_vectors)}")

    return total


def build_qdrant_index(
    data_dir: Path,
    client: Any,
    text_encoder: Any,
    image_encoder: Any,
    limit_per_dataset: int | None = None,
    recreate: bool = False,
    use_patches: bool = True,
) -> dict[str, int]:
    """Full Qdrant indexing pipeline: load → encode → upsert.

    Args:
        data_dir: dataset root directory
        client: Qdrant client
        text_encoder: BioMedBERTEncoder
        image_encoder: BioCLIPEncoder
        limit_per_dataset: optional row limit per dataset
        recreate: drop and recreate collections first
        use_patches: enable 5-patch image splitting

    Returns:
        dict with text_points and image_points counts
    """
    from medical_rag.data_loaders import load_corpora

    logger.info("Loading corpora...")
    documents, images = load_corpora(data_dir, limit_per_dataset=limit_per_dataset)
    logger.info(f"Loaded {len(documents)} documents, {len(images)} images")

    init_qdrant_collections(client, recreate=recreate)

    logger.info("Indexing text chunks...")
    text_points = index_text_chunks(client, documents, text_encoder)

    logger.info("Indexing image patches...")
    image_points = index_image_patches(
        client, images, image_encoder, use_patches=use_patches
    )

    logger.info(f"Indexing complete: {text_points} text points, {image_points} image points")
    return {"text_points": text_points, "image_points": image_points}
