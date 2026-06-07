"""Process PathVQA canonical files into the unified processed JSONL corpus.

This script is intentionally lightweight and resumable-safe: it rewrites only
`data/processed/pathvqa_processed.jsonl` and a small summary file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_image_path(data_dir: Path, value: object) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path)
    prefixed = data_dir / path
    if prefixed.exists():
        return str(prefixed)
    return str(prefixed)


def _iter_pathvqa_rows(data_dir: Path) -> Iterable[dict[str, Any]]:
    dataset_dir = data_dir / "pathvqa"
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for file_info in payload.get("files", []):
            raw_path = Path(str(file_info.get("path", "")))
            path = raw_path if raw_path.is_absolute() else dataset_dir / raw_path.name
            if not path.exists():
                continue
            yield from _read_jsonl(path)
        return

    for path in sorted(dataset_dir.glob("pathvqa_*.jsonl")):
        yield from _read_jsonl(path)


def process_pathvqa(data_dir: Path, output_file: Path, validate_only: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    missing_images = 0
    image_count = 0

    for idx, row in enumerate(_iter_pathvqa_rows(data_dir)):
        split = str(row.get("split") or "unknown")
        record_id = str(row.get("record_id") or f"pathvqa-{split}-{idx}")
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        text = str(row.get("text") or "").strip()
        image_path = _resolve_image_path(data_dir, row.get("image_path"))
        if image_path:
            image_count += 1
            if not Path(image_path).exists():
                missing_images += 1

        processed = {
            "dataset": "pathvqa",
            "split": split,
            "source_id": row.get("source_id") or "pathvqa",
            "record_id": record_id,
            "text": text or "\n".join(part for part in [question, answer] if part),
            "question": question or None,
            "answer": answer or None,
            "image_path": image_path,
            "metadata": {
                "raw": row.get("metadata", {}),
            },
        }
        rows.append(processed)
        split_counts[split] = split_counts.get(split, 0) + 1

    summary = {
        "dataset": "pathvqa",
        "rows": len(rows),
        "splits": split_counts,
        "image_path_count": image_count,
        "missing_images": missing_images,
        "output_file": str(output_file),
        "validate_only": validate_only,
    }

    if not validate_only:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_file = output_file.with_suffix(".summary.json")
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Process PathVQA into data/processed")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-file", default="data/processed/pathvqa_processed.jsonl")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    summary = process_pathvqa(
        data_dir=Path(args.data_dir),
        output_file=Path(args.output_file),
        validate_only=args.validate_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["rows"] == 0:
        raise SystemExit("No PathVQA rows found")
    if summary["missing_images"]:
        raise SystemExit(f"PathVQA has {summary['missing_images']} missing images")


if __name__ == "__main__":
    main()
