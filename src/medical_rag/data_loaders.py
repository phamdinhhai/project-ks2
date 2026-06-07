from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from medical_rag.schema import DocumentChunk, ImageRecord

PROCESSED_FILES = {
    "medqa": "medqa_processed.jsonl",
    "bioasq": "bioasq_processed.jsonl",
    "vqa_rad": "vqa_rad_processed.jsonl",
    "roco": "roco_processed.jsonl",
    "mimic_cxr": "mimic_cxr_processed.jsonl",
    "pathvqa": "pathvqa_processed.jsonl",
}


def _read_jsonl(path: Path, limit: int | None = None) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _option_text(options: object) -> str:
    if not isinstance(options, list):
        return ""
    parts = []
    for option in options:
        if isinstance(option, dict):
            parts.append(f"{option.get('key', '')}. {option.get('value', '')}".strip())
    return "\n".join(parts)


def _resolve_image_path(data_dir: Path, image_path: object) -> str | None:
    if not image_path:
        return None
    path = Path(str(image_path))
    if path.is_absolute():
        return str(path)
    return str(data_dir / path)


def _processed_text(row: dict) -> str:
    text = str(row.get("text") or "").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if text and question and question not in text:
        text = f"Question: {question}\nContext: {text}"
    if answer and answer not in text:
        text = f"{text}\nAnswer: {answer}" if text else f"Answer: {answer}"
    return text or question or answer


def _image_caption(row: dict) -> str:
    dataset = str(row.get("dataset") or "")
    text = str(row.get("text") or "").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if dataset in {"vqa_rad", "pathvqa"}:
        return "\n".join(part for part in [f"Question: {question}" if question else "", f"Answer: {answer}" if answer else "", text] if part)
    if dataset == "mimic_cxr":
        return f"MIMIC-CXR radiology report:\n{text}"
    if dataset == "roco":
        return text
    return _processed_text(row)


def _metadata(row: dict, image_path: str | None = None) -> dict:
    record_id = str(row.get("record_id") or "")
    metadata = dict(row.get("metadata") or {})
    metadata.update({
        "split": row.get("split"),
        "source_id": row.get("source_id"),
        "record_id": record_id,
        "base_record_id": record_id,
        "question": row.get("question"),
        "answer": row.get("answer"),
        "image_path": image_path,
        "image_path_exists": bool(image_path and Path(image_path).exists()),
    })
    return metadata


def load_processed_dataset(data_dir: Path, dataset: str, limit: int | None = None) -> tuple[list[DocumentChunk], list[ImageRecord]]:
    processed_file = data_dir / "processed" / PROCESSED_FILES[dataset]
    if not processed_file.exists():
        return [], []
    documents: list[DocumentChunk] = []
    images: list[ImageRecord] = []
    for row_idx, row in enumerate(_read_jsonl(processed_file, limit=limit)):
        dataset_name = str(row.get("dataset") or dataset)
        record_id = str(row.get("record_id") or f"row-{row_idx}")
        text = _processed_text(row)
        image_path = _resolve_image_path(data_dir, row.get("image_path"))
        metadata = _metadata(row, image_path=image_path)
        if text:
            documents.append(DocumentChunk(
                id=f"{dataset_name}-text-{record_id}",
                text=text,
                dataset=dataset_name,
                source_path=str(processed_file),
                metadata=metadata,
            ))
        if image_path or dataset_name in {"roco", "vqa_rad", "mimic_cxr", "pathvqa"}:
            caption = _image_caption(row)
            if caption:
                images.append(ImageRecord(
                    id=f"{dataset_name}-image-{record_id}",
                    caption=caption,
                    dataset=dataset_name,
                    image_path=image_path,
                    metadata=metadata,
                ))
    return documents, images


def load_medqa(data_dir: Path, limit: int | None = None) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    medqa_dir = data_dir / "medqa"
    for split_path in sorted(medqa_dir.glob("medqa_*.jsonl")):
        split = split_path.stem.replace("medqa_", "")
        for row_idx, row in enumerate(_read_jsonl(split_path, limit=limit)):
            text = (
                f"Question: {row.get('question', '')}\n"
                f"Options:\n{_option_text(row.get('options'))}\n"
                f"Answer: {row.get('answer', '')}"
            )
            record_id = f"{split}-{row_idx}"
            chunks.append(DocumentChunk(
                id=f"medqa-text-{record_id}",
                text=text,
                dataset="medqa",
                source_path=str(split_path),
                metadata={
                    "split": split,
                    "record_id": record_id,
                    "base_record_id": record_id,
                    "answer_idx": row.get("answer_idx"),
                    "answer": row.get("answer"),
                    "meta_info": row.get("meta_info"),
                },
            ))
    return chunks


def load_roco(data_dir: Path, limit: int | None = None) -> list[ImageRecord]:
    roco_file = data_dir / "roco" / "roco_subset_2_5gb.jsonl"
    if not roco_file.exists():
        return []
    records: list[ImageRecord] = []
    image_root = data_dir / "roco" / "hf_subset_2_5gb" / "images"
    for row_idx, row in enumerate(_read_jsonl(roco_file, limit=limit)):
        image_id = str(row.get("image_id") or f"roco-{row_idx}")
        image_path = image_root / f"{image_id}.png"
        records.append(ImageRecord(
            id=f"roco-image-main-{row_idx}",
            caption=str(row.get("caption", "")),
            dataset="roco",
            image_path=str(image_path) if image_path.exists() else None,
            metadata={
                "split": "main",
                "record_id": f"main-{row_idx}",
                "base_record_id": f"main-{row_idx}",
                "source_image_id": image_id,
                "cui": row.get("cui", []),
                "estimated_image_bytes": row.get("estimated_image_bytes"),
                "image_path_exists": image_path.exists(),
            },
        ))
    return records


def load_vqa_rad(data_dir: Path, limit: int | None = None) -> tuple[list[DocumentChunk], list[ImageRecord]]:
    vqa_file = data_dir / "vqa_rad" / "vqa_rad_train.jsonl"
    if not vqa_file.exists() or vqa_file.stat().st_size == 0:
        return [], []
    docs: list[DocumentChunk] = []
    images: list[ImageRecord] = []
    for row_idx, row in enumerate(_read_jsonl(vqa_file, limit=limit)):
        image_name = row.get("image") or row.get("image_name") or row.get("image_id")
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        text = f"Question: {question}\nAnswer: {answer}"
        record_id = f"train-{row_idx}"
        image_path = str(data_dir / "vqa_rad" / "images" / str(image_name)) if image_name else None
        metadata = {"split": "train", "record_id": record_id, "base_record_id": record_id, "question": question, "answer": answer, "image": image_name}
        docs.append(DocumentChunk(id=f"vqa_rad-text-{record_id}", text=text, dataset="vqa_rad", source_path=str(vqa_file), metadata=metadata))
        images.append(ImageRecord(id=f"vqa_rad-image-{record_id}", caption=text, dataset="vqa_rad", image_path=image_path, metadata=metadata))
    return docs, images


def load_mimic_cxr(data_dir: Path, limit: int | None = None) -> list[ImageRecord]:
    image_dir = data_dir / "mimic_cxr" / "images"
    if not image_dir.exists():
        return []
    records: list[ImageRecord] = []
    for idx, image_path in enumerate(image_dir.rglob("*")):
        if limit is not None and idx >= limit:
            break
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        record_id = f"main-{idx}"
        records.append(ImageRecord(
            id=f"mimic_cxr-image-{record_id}",
            caption=f"MIMIC-CXR chest radiograph {image_path.stem}",
            dataset="mimic_cxr",
            image_path=str(image_path),
            metadata={"split": "main", "record_id": record_id, "base_record_id": record_id, "filename": image_path.name, "image_path_exists": True},
        ))
    return records


def load_canonical_dataset(data_dir: Path, dataset: str, limit: int | None = None) -> tuple[list[DocumentChunk], list[ImageRecord]]:
    manifest_file = data_dir / dataset / "manifest.json"
    if not manifest_file.exists():
        return [], []
    
    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    documents: list[DocumentChunk] = []
    images: list[ImageRecord] = []
    
    for file_info in manifest.get("files", []):
        file_path_str = file_info.get("path", "")
        if not file_path_str: continue
        actual_path = data_dir / dataset / Path(file_path_str).name
        if not actual_path.exists(): continue
        
        for row_idx, row in enumerate(_read_jsonl(actual_path, limit=limit)):
            dataset_name = str(row.get("dataset") or dataset)
            record_id = str(row.get("record_id") or f"row-{row_idx}")
            text = _processed_text(row)
            image_path_val = row.get("image_path")
            image_path = _resolve_image_path(data_dir, image_path_val) if image_path_val else None
            metadata = _metadata(row, image_path=image_path)
            
            if text:
                documents.append(DocumentChunk(
                    id=f"{dataset_name}-text-{record_id}",
                    text=text,
                    dataset=dataset_name,
                    source_path=str(actual_path),
                    metadata=metadata,
                ))
            if image_path or dataset_name in {"roco", "vqa_rad", "mimic_cxr", "pathvqa"}:
                caption = _image_caption(row)
                if caption:
                    images.append(ImageRecord(
                        id=f"{dataset_name}-image-{record_id}",
                        caption=caption,
                        dataset=dataset_name,
                        image_path=image_path,
                        metadata=metadata,
                    ))
    return documents, images


def load_corpora(data_dir: Path, limit_per_dataset: int | None = None) -> tuple[list[DocumentChunk], list[ImageRecord]]:
    """Load local corpora with canonical JSONL priority and raw fallbacks."""
    from medical_rag.data_tools.canonicalize import CANONICAL_DATASETS
    documents: list[DocumentChunk] = []
    images: list[ImageRecord] = []

    # First, try to load all defined datasets from their unified canonical format
    for dataset in CANONICAL_DATASETS:
        docs, imgs = load_canonical_dataset(data_dir, dataset, limit=limit_per_dataset)
        if docs or imgs:
            documents.extend(docs)
            images.extend(imgs)
            continue
            
        # Fallback to legacy processed files if canonical missing
        if dataset in PROCESSED_FILES:
            processed_docs, processed_images = load_processed_dataset(data_dir, dataset, limit=limit_per_dataset)
            if processed_docs or processed_images:
                documents.extend(processed_docs)
                images.extend(processed_images)
                continue
                
        # Final raw fallbacks
        if dataset == "medqa":
            documents.extend(load_medqa(data_dir, limit=limit_per_dataset))
        elif dataset == "vqa_rad":
            vqa_docs, vqa_images = load_vqa_rad(data_dir, limit=limit_per_dataset)
            documents.extend(vqa_docs)
            images.extend(vqa_images)
        elif dataset == "roco":
            images.extend(load_roco(data_dir, limit=limit_per_dataset))
        elif dataset == "mimic_cxr":
            images.extend(load_mimic_cxr(data_dir, limit=limit_per_dataset))
            
    return documents, images
