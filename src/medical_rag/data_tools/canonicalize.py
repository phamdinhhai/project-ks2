"""Dataset canonicalization utilities.

Convert local HuggingFace ``save_to_disk`` datasets, plain JSON/JSONL files, and
image metadata folders into transparent JSONL + manifest outputs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CANONICAL_DATASETS = ("medqa", "bioasq", "vqa_rad", "roco", "mimic_cxr", "pathvqa")
CANONICAL_FIELDS = ("dataset", "split", "record_id", "question", "answer", "text", "image_path", "metadata")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hf_dataset(path: Path):
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover - depends on optional local env
        raise RuntimeError("Install HuggingFace datasets first: pip install datasets") from exc
    return load_from_disk(str(path))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(item) for item in value if _as_text(item)).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return None


def _normalize_record(dataset: str, split: str, idx: int, row: dict[str, Any], image_path: str | None = None) -> dict[str, Any]:
    record_id = _as_text(_first(row, ("record_id", "id", "qid", "question_id", "image_id", "dicom_id"))) or f"{dataset}-{split}-{idx}"
    question = _as_text(_first(row, ("question", "query", "body", "prompt", "caption")))
    answer = _as_text(_first(row, ("answer", "answers", "label", "target", "final_decision", "ideal_answer")))
    text = _as_text(_first(row, ("text", "context", "caption", "report", "findings", "impression", "abstract")))
    if not text:
        text = "\n".join(part for part in (question, answer) if part)
    
    # Exclude common image keys and already processed fields from metadata
    exclude_keys = {
        "question", "query", "body", "prompt", 
        "answer", "answers", "label", "target", "final_decision", "ideal_answer",
        "text", "context", "caption", "report", "findings", "impression", "abstract",
        "image", "image_path", "path", "jpg_path"
    }
    
    metadata = {}
    for key, value in row.items():
        if key in exclude_keys:
            continue
        # Skip PIL images or other objects that are not JSON serializable
        if hasattr(value, "save") or "Image" in str(type(value)):
            continue
        metadata[key] = value

    return {
        "dataset": dataset,
        "split": split,
        "record_id": record_id,
        "question": question,
        "answer": answer,
        "text": text,
        "image_path": image_path,
        "metadata": metadata,
    }


def _dataset_splits(ds: Any) -> dict[str, Any]:
    if hasattr(ds, "keys") and not hasattr(ds, "features"):
        return {str(key): ds[key] for key in ds.keys()}
    return {"train": ds}


def _copy_image(value: Any, dataset_dir: Path, split: str, record_id: str, idx: int) -> str | None:
    if value in (None, ""):
        return None
    image_dir = dataset_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (str, Path)):
        source = Path(value)
        if source.exists():
            suffix = source.suffix or ".jpg"
            dest = image_dir / f"{record_id}{suffix}"
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            return str(dest.relative_to(dataset_dir.parent)).replace("\\", "/")
        return str(value).replace("\\", "/")
    if hasattr(value, "save"):
        dest = image_dir / f"{record_id or idx}.jpg"
        value.save(dest)
        return str(dest.relative_to(dataset_dir.parent)).replace("\\", "/")
    return None


def canonicalize_hf_disk(dataset: str, source_dir: Path, output_dir: Path, limit: int | None = None) -> dict[str, Any]:
    ds = load_hf_dataset(source_dir)
    split_counts: dict[str, int] = {}
    files: list[dict[str, Any]] = []
    for split, split_ds in _dataset_splits(ds).items():
        out_file = output_dir / f"{dataset}_{split}.jsonl"
        rows = []
        for idx, row in enumerate(split_ds):
            if limit is not None and idx >= limit:
                break
            row = dict(row)
            image_value = _first(row, ("image", "image_path", "path", "jpg_path"))
            provisional_id = _as_text(_first(row, ("record_id", "id", "qid", "question_id", "image_id", "dicom_id"))) or f"{dataset}-{split}-{idx}"
            image_path = _copy_image(image_value, output_dir, split, provisional_id, idx)
            rows.append(_normalize_record(dataset, split, idx, row, image_path=image_path))
        count = write_jsonl(out_file, rows)
        split_counts[split] = count
        files.append({"path": str(out_file), "rows": count, "sha256": file_sha256(out_file)})
    return write_manifest(dataset, output_dir, "hf_save_to_disk", str(source_dir), split_counts, files)


def canonicalize_medqa(data_dir: Path, limit: int | None = None) -> dict[str, Any]:
    dataset_dir = data_dir / "medqa"
    split_counts: dict[str, int] = {}
    files: list[dict[str, Any]] = []
    source_files = [path for path in sorted(dataset_dir.glob("medqa_*.jsonl")) if ".canonical" not in path.name]
    for path in source_files:
        split = path.stem.replace("medqa_", "")
        rows = (_normalize_record("medqa", split, idx, row) for idx, row in enumerate(read_jsonl(path)) if limit is None or idx < limit)
        out_file = dataset_dir / f"medqa_{split}.canonical.jsonl"
        count = write_jsonl(out_file, rows)
        split_counts[split] = count
        files.append({"path": str(out_file), "rows": count, "sha256": file_sha256(out_file)})
    return write_manifest("medqa", dataset_dir, "local_jsonl", str(dataset_dir), split_counts, files)


def canonicalize_roco(data_dir: Path, limit: int | None = None) -> dict[str, Any]:
    dataset_dir = data_dir / "roco"
    # Try multiple metadata sources
    meta_sources = ["metadata.jsonl", "roco_subset_2_5gb.jsonl"]
    meta = None
    for src in meta_sources:
        candidate = dataset_dir / src
        if candidate.exists():
            meta = candidate
            break

    if meta:
        rows = []
        for idx, row in enumerate(read_jsonl(meta)):
            if limit is not None and idx >= limit:
                break
            split = _as_text(row.get("split")) or "train"
            # ROCO specific: if image_path is missing but image_id exists, 
            # try to construct path if images exist locally
            image_path = _as_text(_first(row, ("image_path", "path")))
            if not image_path and "image_id" in row:
                image_id = row["image_id"]
                # Look for image in standard locations
                for suffix in [".jpg", ".png", ".jpeg"]:
                    img_cand = dataset_dir / "images" / split / f"{image_id}{suffix}"
                    if img_cand.exists():
                        image_path = str(img_cand.relative_to(data_dir)).replace("\\", "/")
                        break
            
            rows.append(_normalize_record("roco", split, idx, row, image_path=image_path))
        
        by_split: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_split.setdefault(row["split"], []).append(row)
        
        files = []
        split_counts = {}
        for split, split_rows in by_split.items():
            out_file = dataset_dir / f"roco_{split}.jsonl"
            count = write_jsonl(out_file, split_rows)
            split_counts[split] = count
            files.append({"path": str(out_file), "rows": count, "sha256": file_sha256(out_file)})
        return write_manifest("roco", dataset_dir, "local_metadata_jsonl", str(meta), split_counts, files)
    
    # Fallback to HF disk if it exists
    hf_source = dataset_dir / "hf_subset_2_5gb"
    if hf_source.exists() and any(hf_source.iterdir()):
        try:
            return canonicalize_hf_disk("roco", hf_source, dataset_dir, limit=limit)
        except Exception:
            # If HF load fails, and we already tried JSONL, we might be in trouble
            pass

    raise FileNotFoundError(f"No valid ROCO source found in {dataset_dir}")


def canonicalize_dataset(dataset: str, data_dir: Path, limit: int | None = None) -> dict[str, Any]:
    dataset = dataset.lower()
    if dataset == "medqa":
        return canonicalize_medqa(data_dir, limit=limit)
    if dataset == "roco":
        return canonicalize_roco(data_dir, limit=limit)

    source = data_dir / dataset / "hf_dataset"
    if not source.exists():
        raise FileNotFoundError(f"Missing source for {dataset}: {source}")
    return canonicalize_hf_disk(dataset, source, data_dir / dataset, limit=limit)


def canonicalize_all(data_dir: Path, datasets: Iterable[str] = CANONICAL_DATASETS, limit: int | None = None) -> dict[str, Any]:
    results = {}
    for dataset in datasets:
        results[dataset] = canonicalize_dataset(dataset, data_dir, limit=limit)
    return {"generated_at": utc_now(), "data_dir": str(data_dir), "datasets": results}


def write_manifest(dataset: str, output_dir: Path, source_type: str, source: str, split_counts: dict[str, int], files: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "dataset": dataset,
        "generated_at": utc_now(),
        "source_type": source_type,
        "source": source,
        "canonical_fields": list(CANONICAL_FIELDS),
        "split_counts": split_counts,
        "total_rows": sum(split_counts.values()),
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def audit_canonical_dataset(dataset_dir: Path, dataset: str) -> dict[str, Any]:
    files = sorted(dataset_dir.glob(f"{dataset}_*.jsonl")) + sorted(dataset_dir.glob(f"{dataset}_*.canonical.jsonl"))
    seen: Counter[str] = Counter()
    file_reports = []
    for path in files:
        rows = 0
        image_paths = 0
        existing_images = 0
        for row in read_jsonl(path):
            rows += 1
            rid = str(row.get("record_id") or "")
            if rid:
                seen[rid] += 1
            image_path = row.get("image_path")
            if image_path:
                image_paths += 1
                candidate = Path(str(image_path))
                if not candidate.is_absolute():
                    candidate = dataset_dir.parent / candidate
                existing_images += int(candidate.exists())
        file_reports.append({"path": str(path), "rows": rows, "image_path_count": image_paths, "image_path_existing": existing_images})
    duplicates = [rid for rid, count in seen.items() if count > 1]
    return {"exists": dataset_dir.exists(), "files": file_reports, "duplicate_record_ids": duplicates[:50], "duplicate_count": len(duplicates)}
