from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from medical_rag.data_tools.canonicalize import CANONICAL_DATASETS, audit_canonical_dataset

PROCESSED_FILES = {
    "medqa": "medqa_processed.jsonl",
    "bioasq": "bioasq_processed.jsonl",
    "vqa_rad": "vqa_rad_processed.jsonl",
    "roco": "roco_processed.jsonl",
    "mimic_cxr": "mimic_cxr_processed.jsonl",
}
ROUTING = {
    "medqa": "text-only",
    "bioasq": "text-only",
    "vqa_rad": "text + image",
    "roco": "image/caption",
    "mimic_cxr": "report text + image",
}


def _iter_jsonl(path: Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _safe_exists(path_value: str | None) -> bool:
    if not path_value:
        return False
    return Path(path_value).exists()


def audit_processed_file(path: Path, sample_size: int = 2) -> dict[str, Any]:
    counts = Counter()
    samples = []
    row_count = 0
    image_path_count = 0
    image_path_existing = 0
    for row in _iter_jsonl(path):
        row_count += 1
        dataset = str(row.get("dataset") or "unknown")
        split = str(row.get("split") or "unknown")
        counts[f"split:{split}"] += 1
        image_path = row.get("image_path")
        if image_path:
            image_path_count += 1
            image_path_existing += int(_safe_exists(str(image_path)))
        if len(samples) < sample_size:
            samples.append({
                "dataset": dataset,
                "split": split,
                "record_id": row.get("record_id"),
                "has_text": bool(row.get("text")),
                "has_question": bool(row.get("question")),
                "has_answer": bool(row.get("answer")),
                "has_image_path": bool(image_path),
                "image_path_exists": _safe_exists(str(image_path)) if image_path else False,
                "text_preview": str(row.get("text") or "")[:180],
            })
    return {
        "exists": True,
        "path": str(path),
        "rows": row_count,
        "splits": dict(counts),
        "image_path_count": image_path_count,
        "image_path_existing": image_path_existing,
        "samples": samples,
    }


def audit_data_directory(data_dir: Path, sample_size: int = 2) -> dict[str, Any]:
    processed_dir = data_dir / "processed"
    datasets: dict[str, Any] = {}
    
    # Audit all known canonical datasets
    for dataset in CANONICAL_DATASETS:
        dataset_dir = data_dir / dataset
        entry: dict[str, Any] = {
            "dataset": dataset,
            "folder_exists": dataset_dir.exists(),
        }
        
        if dataset in ROUTING:
            entry["recommended_branch"] = ROUTING[dataset]
            
        # 1. Audit canonical raw files
        if dataset_dir.exists():
            entry["canonical_raw"] = audit_canonical_dataset(dataset_dir, dataset)
            
        # 2. Audit processed indexable files
        if dataset in PROCESSED_FILES:
            processed_file = processed_dir / PROCESSED_FILES[dataset]
            entry["processed_file"] = str(processed_file)
            entry["processed_exists"] = processed_file.exists()
            if processed_file.exists():
                entry["processed"] = audit_processed_file(processed_file, sample_size=sample_size)
                
        datasets[dataset] = entry

    raw_pdf_dir = data_dir / "raw_pdfs"
    return {
        "data_dir": str(data_dir),
        "processed_dir": str(processed_dir),
        "datasets": datasets,
        "raw_pdfs": {
            "folder_exists": raw_pdf_dir.exists(),
            "status": "excluded",
            "used_for_indexing": False,
            "message": "Temporarily excluded by project scope; only non-PDF dataset folders are used.",
        },
    }
