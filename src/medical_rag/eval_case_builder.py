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


def _build_text_case(dataset: str, record_id: str, row: dict[str, Any]) -> dict[str, Any]:
    query = _query_text(row, dataset)
    return {
        "query": query,
        "gold_text_id": f"{dataset}-text-{record_id}",
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
    return {
        "query": query,
        "gold_image_id": f"{dataset}-image-{record_id}",
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
        path = processed_dir / PROCESSED_FILES[dataset]
        if not path.exists():
            summary["datasets"][dataset] = {
                "exists": False,
                "target": target,
                "created": 0,
                "text_cases": 0,
                "image_cases": 0,
            }
            continue

        text_cases = 0
        image_cases = 0
        created = 0
        dataset_rows = list(_iter_jsonl(path))
        if not dataset_rows:
            summary["datasets"][dataset] = {
                "exists": True,
                "target": target,
                "rows": 0,
                "created": 0,
                "text_cases": 0,
                "image_cases": 0,
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
            "source_file": str(path),
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
