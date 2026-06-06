from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TARGET_CASES = {
    "medqa": 40,
    "bioasq": 40,
    "vqa_rad": 60,
    "roco": 30,
    "mimic_cxr": 30,
}
PROCESSED_FILES = {
    "medqa": "medqa_processed.jsonl",
    "bioasq": "bioasq_processed.jsonl",
    "vqa_rad": "vqa_rad_processed.jsonl",
    "roco": "roco_processed.jsonl",
    "mimic_cxr": "mimic_cxr_processed.jsonl",
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _canonical_files(data_dir: Path, dataset: str) -> list[Path]:
    """Return canonical JSONL files for a dataset, matching data_loaders priority."""
    manifest_file = data_dir / dataset / "manifest.json"
    if not manifest_file.exists():
        return []
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    files: list[Path] = []
    for file_info in manifest.get("files", []):
        file_path_str = str(file_info.get("path") or "")
        if not file_path_str:
            continue
        # data_loaders.load_canonical_dataset resolves by basename under data/<dataset>.
        actual_path = data_dir / dataset / Path(file_path_str).name
        if actual_path.exists():
            files.append(actual_path)
    return files


def _load_dataset_rows(data_dir: Path, dataset: str) -> tuple[list[dict[str, Any]], str | None, bool]:
    """Load rows from canonical files when available, otherwise legacy processed JSONL."""
    canonical_paths = _canonical_files(data_dir, dataset)
    if canonical_paths:
        rows: list[dict[str, Any]] = []
        for path in canonical_paths:
            rows.extend(_iter_jsonl(path))
        source = ",".join(str(path) for path in canonical_paths)
        return rows, source, True

    processed_path = data_dir / "processed" / PROCESSED_FILES[dataset]
    if processed_path.exists():
        return list(_iter_jsonl(processed_path)), str(processed_path), False
    return [], None, False

def _language_of(text: str) -> str:
    lower = text.lower()
    vi_chars = "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    if any(char in lower for char in vi_chars):
        return "vi"
    return "en"


def _query_text(row: dict[str, Any], dataset: str) -> str:
    question = str(row.get("question") or "").strip()
    text = str(row.get("text") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if question:
        return question
    if dataset == "roco":
        return f"Find matching medical image: {text}" if text else "find matching medical image"
    if text:
        return text.split("\n")[0][:220]
    return answer[:220] if answer else f"{dataset} sample"


def _canonical_record_id(dataset: str, record_id: str) -> str:
    """Normalize processed record_id to canonical/index record_id format.

    Canonicalized datasets prefix record IDs with the dataset name, e.g.
    processed `train-0` becomes indexed as `medqa-train-0`.
    If the prefix is already present, keep it unchanged.
    """
    record_id = str(record_id or "")
    if dataset == "roco":
        return record_id
    return record_id if record_id.startswith(f"{dataset}-") else f"{dataset}-{record_id}"


def _build_text_case(dataset: str, record_id: str, row: dict[str, Any]) -> dict[str, Any]:
    query = _query_text(row, dataset)
    canonical_id = _canonical_record_id(dataset, record_id)
    return {
        "query": query,
        "gold_text_id": f"{dataset}-text-{canonical_id}",
        "gold_answer": row.get("answer"),
        "query_language": _language_of(query),
        "modality": "text",
        "dataset_hint": dataset,
        "dataset": dataset,
        "task_type": "qa_text",
    }


def _build_image_case(dataset: str, record_id: str, row: dict[str, Any]) -> dict[str, Any]:
    query = _query_text(row, dataset)
    task_type = "qa_image" if dataset in {"vqa_rad", "mimic_cxr"} else "caption_image"
    canonical_id = _canonical_record_id(dataset, record_id)
    return {
        "query": query,
        "gold_image_id": f"{dataset}-image-{canonical_id}",
        "gold_answer": row.get("answer"),
        "query_language": _language_of(query),
        "modality": "image",
        "dataset_hint": dataset,
        "dataset": dataset,
        "task_type": task_type,
    }


def build_eval_cases(
    data_dir: Path,
    targets: dict[str, int] | None = None,
    include_mixed_for_vqa_rad: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processed_dir = data_dir / "processed"
    targets = targets or TARGET_CASES
    cases: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "data_dir": str(data_dir),
        "processed_dir": str(processed_dir),
        "targets": targets,
        "datasets": {},
        "raw_pdfs_used": False,
        "excluded_sources": ["raw_pdfs"],
    }

    for dataset, target in targets.items():
        dataset_rows, source_file, used_canonical = _load_dataset_rows(data_dir, dataset)
        if not dataset_rows:
            summary["datasets"][dataset] = {
                "exists": False,
                "target": target,
                "created": 0,
                "text_cases": 0,
                "image_cases": 0,
                "used_canonical": used_canonical,
                "source_file": source_file,
            }
            continue

        text_cases = 0
        image_cases = 0
        created = 0
        if not dataset_rows:
            summary["datasets"][dataset] = {
                "exists": True,
                "target": target,
                "rows": 0,
                "created": 0,
                "text_cases": 0,
                "image_cases": 0,
                "used_canonical": used_canonical,
                "source_file": source_file,
            }
            continue

        step = max(1, len(dataset_rows) // max(1, target))
        selected_rows = dataset_rows[::step][:target]

        for row in selected_rows:
            record_id = str(row.get("record_id") or "")
            if not record_id:
                continue

            if dataset in {"medqa", "bioasq"}:
                cases.append(_build_text_case(dataset, record_id, row))
                text_cases += 1
                created += 1
            elif dataset == "roco":
                cases.append(_build_image_case(dataset, record_id, row))
                image_cases += 1
                created += 1
            elif dataset == "vqa_rad":
                cases.append(_build_image_case(dataset, record_id, row))
                image_cases += 1
                created += 1
                if include_mixed_for_vqa_rad and created < target:
                    cases.append(_build_text_case(dataset, record_id, row))
                    text_cases += 1
                    created += 1
            elif dataset == "mimic_cxr":
                # Alternate between text and image cases for coverage.
                if created % 2 == 0:
                    cases.append(_build_text_case(dataset, record_id, row))
                    text_cases += 1
                else:
                    cases.append(_build_image_case(dataset, record_id, row))
                    image_cases += 1
                created += 1

            if created >= target:
                break

        summary["datasets"][dataset] = {
            "exists": True,
            "target": target,
            "rows": len(dataset_rows),
            "created": created,
            "text_cases": text_cases,
            "image_cases": image_cases,
            "used_canonical": used_canonical,
            "source_file": source_file,
        }

    summary["total_cases"] = len(cases)
    summary["text_cases"] = sum(1 for case in cases if case.get("modality") == "text")
    summary["image_cases"] = sum(1 for case in cases if case.get("modality") == "image")
    return cases, summary


def write_eval_cases(cases: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


def write_eval_summary(summary: dict[str, Any], summary_file: Path) -> None:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
