"""Resumable production Qdrant indexing.

This module is designed for Colab Free runtimes: it uses deterministic point IDs,
flushes a progress state after each successful batch, and supports time/point
budgets so indexing can continue across multiple sessions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from medical_rag.ingestion.indexer import IMAGE_DIM, TEXT_DIM

logger = logging.getLogger(__name__)

DEFAULT_TEXT_COLLECTION = "text_chunks_prod"
DEFAULT_IMAGE_COLLECTION = "image_patches_prod"
DEFAULT_TEXT_MODEL = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
DEFAULT_IMAGE_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

IndexModality = Literal["text", "image", "both"]
ImageMode = Literal["full_only", "patches", "caption_fallback"]


@dataclass
class ResumableIndexConfig:
    data_dir: Path
    state_file: Path = Path("outputs/index_state/full_index_state.json")
    datasets: list[str] = field(default_factory=list)
    modality: IndexModality = "both"
    image_mode: ImageMode = "full_only"
    text_collection: str = DEFAULT_TEXT_COLLECTION
    image_collection: str = DEFAULT_IMAGE_COLLECTION
    batch_size: int = 16
    max_records: int | None = None
    max_text_points: int | None = None
    max_image_points: int | None = None
    max_minutes: float | None = None
    dry_run: bool = False
    recreate: bool = False
    text_model_name: str = DEFAULT_TEXT_MODEL
    image_model_name: str = DEFAULT_IMAGE_MODEL


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_uuid(source: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source))


def _model_version(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "completed": {"text": [], "image": []},
            "failed": {"text": [], "image": []},
            "counters": {"text_points": 0, "image_points": 0},
            "runs": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_completed_set(state: dict[str, Any], modality: str) -> set[str]:
    return set(str(item) for item in state.get("completed", {}).get(modality, []))


def _mark_completed(state: dict[str, Any], modality: str, ids: Iterable[str]) -> None:
    completed = state.setdefault("completed", {}).setdefault(modality, [])
    existing = set(completed)
    for item in ids:
        if item not in existing:
            completed.append(item)
            existing.add(item)


def _mark_failed(state: dict[str, Any], modality: str, source_id: str, error: Exception) -> None:
    state.setdefault("failed", {}).setdefault(modality, []).append({
        "source_id": source_id,
        "error": repr(error),
        "time": _utc_now(),
    })


def _init_collection(client: Any, collection_name: str, dim: int, recreate: bool = False) -> None:
    from qdrant_client.models import Distance, VectorParams

    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name in existing:
        if recreate:
            client.delete_collection(collection_name=collection_name)
        else:
            return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def _collection_count(client: Any, collection_name: str) -> int | None:
    try:
        result = client.count(collection_name=collection_name, exact=True)
        return int(result.count)
    except Exception:
        return None


def _selected_datasets(datasets: list[str]) -> list[str] | None:
    if not datasets or datasets == ["all"]:
        return None
    return datasets


def _load_selected_corpora(data_dir: Path, datasets: list[str], limit: int | None) -> tuple[list[Any], list[Any]]:
    from medical_rag.data_loaders import load_corpora, load_processed_dataset
    from medical_rag.data_tools.canonicalize import CANONICAL_DATASETS

    selected = _selected_datasets(datasets)
    if selected is None:
        return load_corpora(data_dir, limit_per_dataset=limit)

    documents: list[Any] = []
    images: list[Any] = []
    valid = set(CANONICAL_DATASETS)
    for dataset in selected:
        if dataset not in valid:
            raise ValueError(f"Unknown dataset: {dataset}")
        docs, imgs = load_processed_dataset(data_dir, dataset, limit=limit)
        if not docs and not imgs:
            from medical_rag.data_loaders import load_canonical_dataset
            docs, imgs = load_canonical_dataset(data_dir, dataset, limit=limit)
        documents.extend(docs)
        images.extend(imgs)
    return documents, images


def _text_payload(doc: Any, chunk_idx: int, chunk_text: str) -> dict[str, Any]:
    return {
        "doc_id": doc.id,
        "chunk_idx": chunk_idx,
        "text": chunk_text,
        "dataset": doc.dataset,
        "source_path": doc.source_path,
        "record_id": doc.metadata.get("record_id", ""),
        "base_record_id": doc.metadata.get("base_record_id", ""),
        "question": doc.metadata.get("question", ""),
        "answer": doc.metadata.get("answer", ""),
    }


def _prepare_text_items(documents: list[Any], model_version: str) -> list[dict[str, Any]]:
    from medical_rag.ingestion.chunker import chunk_text

    items: list[dict[str, Any]] = []
    for doc in documents:
        for chunk_idx, text in enumerate(chunk_text(doc.text, window_size=3, overlap=1)):
            source_id = f"text:{doc.dataset}:{doc.metadata.get('record_id', doc.id)}:{chunk_idx}:{model_version}"
            items.append({
                "source_id": source_id,
                "point_id": _stable_uuid(source_id),
                "text": text,
                "payload": _text_payload(doc, chunk_idx, text),
            })
    return items


def _iter_image_items(image_record: Any, image_mode: ImageMode, model_version: str) -> Iterable[dict[str, Any]]:
    record_id = image_record.metadata.get("record_id", image_record.id)
    base_payload = {
        "image_id": image_record.id,
        "caption": image_record.caption,
        "dataset": image_record.dataset,
        "image_path": image_record.image_path,
        "record_id": record_id,
        "base_record_id": image_record.metadata.get("base_record_id", ""),
    }

    if image_mode == "caption_fallback" or not image_record.image_path:
        patch_id = "caption_fallback"
        source_id = f"image:{image_record.dataset}:{record_id}:{patch_id}:{model_version}"
        yield {
            "source_id": source_id,
            "point_id": _stable_uuid(source_id),
            "mode": "caption",
            "caption": image_record.caption or "medical image",
            "payload": {**base_payload, "patch_id": patch_id, "bbox": [0.0, 0.0, 1.0, 1.0]},
        }
        return

    path = Path(image_record.image_path)
    if image_mode == "full_only":
        source_id = f"image:{image_record.dataset}:{record_id}:full:{model_version}"
        yield {
            "source_id": source_id,
            "point_id": _stable_uuid(source_id),
            "mode": "image_path",
            "image_path": str(path),
            "payload": {**base_payload, "patch_id": "full", "bbox": [0.0, 0.0, 1.0, 1.0]},
        }
        return

    from medical_rag.ingestion.chunker import chunk_image

    for patch in chunk_image(path):
        patch_id = str(patch["patch_id"])
        source_id = f"image:{image_record.dataset}:{record_id}:{patch_id}:{model_version}"
        yield {
            "source_id": source_id,
            "point_id": _stable_uuid(source_id),
            "mode": "pil_image",
            "image": patch["image"],
            "payload": {**base_payload, "patch_id": patch_id, "bbox": list(patch["bbox"])},
        }


def _time_exceeded(start_time: float, max_minutes: float | None) -> bool:
    if max_minutes is None:
        return False
    return (time.monotonic() - start_time) >= max_minutes * 60.0


def run_resumable_index(
    client: Any,
    text_encoder: Any,
    image_encoder: Any,
    config: ResumableIndexConfig,
) -> dict[str, Any]:
    from qdrant_client.models import PointStruct

    start_time = time.monotonic()
    state = _load_state(config.state_file)
    state["config"] = {
        "datasets": config.datasets or ["all"],
        "modality": config.modality,
        "image_mode": config.image_mode,
        "text_collection": config.text_collection,
        "image_collection": config.image_collection,
        "text_model_name": config.text_model_name,
        "image_model_name": config.image_model_name,
    }
    state.setdefault("runs", []).append({"started_at": _utc_now(), "dry_run": config.dry_run})

    if not config.dry_run:
        if config.modality in {"text", "both"}:
            _init_collection(client, config.text_collection, TEXT_DIM, recreate=config.recreate)
        if config.modality in {"image", "both"}:
            _init_collection(client, config.image_collection, IMAGE_DIM, recreate=config.recreate)

    documents, images = _load_selected_corpora(config.data_dir, config.datasets, config.max_records)
    text_version = _model_version(config.text_model_name)
    image_version = _model_version(config.image_model_name)

    completed_text = _as_completed_set(state, "text")
    completed_image = _as_completed_set(state, "image")

    summary: dict[str, Any] = {
        "documents_loaded": len(documents),
        "images_loaded": len(images),
        "text_points_attempted": 0,
        "text_points_indexed": 0,
        "image_points_attempted": 0,
        "image_points_indexed": 0,
        "dry_run": config.dry_run,
        "state_file": str(config.state_file),
    }

    if config.modality in {"text", "both"}:
        text_items = [item for item in _prepare_text_items(documents, text_version) if item["source_id"] not in completed_text]
        summary["text_points_pending"] = len(text_items)
        if config.dry_run:
            summary["text_points_attempted"] = len(text_items)
        else:
            for start in range(0, len(text_items), config.batch_size):
                if _time_exceeded(start_time, config.max_minutes):
                    break
                if config.max_text_points is not None and summary["text_points_indexed"] >= config.max_text_points:
                    break
                batch = text_items[start : start + config.batch_size]
                if config.max_text_points is not None:
                    remaining = config.max_text_points - summary["text_points_indexed"]
                    batch = batch[:remaining]
                vectors = text_encoder.encode_batch([item["text"] for item in batch], normalize=True)
                points = [
                    PointStruct(id=item["point_id"], vector=vectors[idx].tolist(), payload=item["payload"])
                    for idx, item in enumerate(batch)
                ]
                client.upsert(collection_name=config.text_collection, points=points)
                ids = [item["source_id"] for item in batch]
                _mark_completed(state, "text", ids)
                summary["text_points_attempted"] += len(batch)
                summary["text_points_indexed"] += len(batch)
                state.setdefault("counters", {})["text_points"] = len(state["completed"]["text"])
                _save_state(config.state_file, state)
                print(f"[text] indexed {summary['text_points_indexed']} this run; total completed={state['counters']['text_points']}", flush=True)

    if config.modality in {"image", "both"}:
        indexed_this_run = 0
        for image_record in images:
            if _time_exceeded(start_time, config.max_minutes):
                break
            try:
                items = [item for item in _iter_image_items(image_record, config.image_mode, image_version) if item["source_id"] not in completed_image]
            except Exception as exc:
                _mark_failed(state, "image", image_record.id, exc)
                _save_state(config.state_file, state)
                continue
            if config.dry_run:
                summary["image_points_attempted"] += len(items)
                continue
            for start in range(0, len(items), config.batch_size):
                if _time_exceeded(start_time, config.max_minutes):
                    break
                if config.max_image_points is not None and indexed_this_run >= config.max_image_points:
                    break
                batch = items[start : start + config.batch_size]
                if config.max_image_points is not None:
                    remaining = config.max_image_points - indexed_this_run
                    batch = batch[:remaining]
                vectors: list[Any] = []
                for item in batch:
                    try:
                        if item["mode"] == "caption":
                            vectors.append(image_encoder.encode_text(item["caption"]))
                        elif item["mode"] == "image_path":
                            from PIL import Image
                            with Image.open(item["image_path"]) as image:
                                vectors.append(image_encoder.encode_image(image.convert("RGB")))
                        else:
                            vectors.append(image_encoder.encode_image(item["image"]))
                    except Exception as exc:
                        _mark_failed(state, "image", item["source_id"], exc)
                        vectors.append(image_encoder.encode_text(item["payload"].get("caption") or "medical image"))
                        item["payload"]["patch_id"] = "caption_fallback"
                        item["payload"]["indexing_fallback"] = "caption_text"
                points = [
                    PointStruct(id=item["point_id"], vector=vectors[idx].tolist(), payload=item["payload"])
                    for idx, item in enumerate(batch)
                ]
                client.upsert(collection_name=config.image_collection, points=points)
                ids = [item["source_id"] for item in batch]
                _mark_completed(state, "image", ids)
                completed_image.update(ids)
                indexed_this_run += len(batch)
                summary["image_points_attempted"] += len(batch)
                summary["image_points_indexed"] += len(batch)
                state.setdefault("counters", {})["image_points"] = len(state["completed"]["image"])
                _save_state(config.state_file, state)
                print(f"[image] indexed {summary['image_points_indexed']} this run; total completed={state['counters']['image_points']}", flush=True)
            if config.max_image_points is not None and indexed_this_run >= config.max_image_points:
                break

    state["runs"][-1]["finished_at"] = _utc_now()
    state["runs"][-1]["summary"] = summary
    _save_state(config.state_file, state)

    summary["qdrant_counts"] = {
        "text": None if config.dry_run else _collection_count(client, config.text_collection),
        "image": None if config.dry_run else _collection_count(client, config.image_collection),
    }
    return summary
