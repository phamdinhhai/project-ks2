"""Data leak check — verify eval cases do not overlap with index.

Usage:
    python scripts/check_data_leak.py
    python scripts/check_data_leak.py --eval-file data/eval_cases_final.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Ensure package importable when run directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def check_data_leak(
    eval_file: Path,
    index_dir: Path,
    verbose: bool = True,
) -> dict:
    """Check for overlap between eval gold IDs and indexed document/image IDs.

    Returns:
        dict with leak_count, total_cases, leak_ids, safe
    """
    from medical_rag.indexing import load_indexes

    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    bundle = load_indexes(index_dir)

    # Collect all indexed IDs
    doc_ids: set[str] = {doc.id for doc in bundle.get("documents", [])}
    img_ids: set[str] = {img.id for img in bundle.get("images", [])}
    alias_map: dict[str, list[str]] = bundle.get("id_aliases", {})

    all_indexed_ids = doc_ids | img_ids

    leak_ids: list[str] = []
    dataset_leak_counts: Counter[str] = Counter()

    for case in cases:
        gold = case.get("gold_text_id") or case.get("gold_image_id")
        if not gold:
            continue

        # Direct match
        if gold in all_indexed_ids:
            leak_ids.append(gold)
            dataset_leak_counts[case.get("dataset", "unknown")] += 1
            continue

        # Alias match
        canonical_ids = set(alias_map.get(gold, []))
        if canonical_ids & all_indexed_ids:
            leak_ids.append(gold)
            dataset_leak_counts[case.get("dataset", "unknown")] += 1

    total = len(cases)
    leak_count = len(leak_ids)
    safe = leak_count == 0

    result = {
        "eval_file": str(eval_file),
        "index_dir": str(index_dir),
        "total_cases": total,
        "indexed_docs": len(doc_ids),
        "indexed_images": len(img_ids),
        "leak_count": leak_count,
        "leak_rate": round(leak_count / total, 4) if total else 0.0,
        "safe": safe,
        "leaking_datasets": dict(dataset_leak_counts),
        "leak_ids_sample": leak_ids[:20],
    }

    if verbose:
        status = "[CLEAN] No data leak detected." if safe else f"[WARNING] DATA LEAK — {leak_count}/{total} cases overlap index"
        print(f"\n{status}")
        print(json.dumps({k: v for k, v in result.items() if k != "leak_ids_sample"}, indent=2))
        if leak_ids:
            print(f"\nSample leaked IDs (first 20):")
            for lid in leak_ids[:20]:
                print(f"  - {lid}")

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check data leak between eval cases and index")
    parser.add_argument("--eval-file", default="data/eval_cases_final.json")
    parser.add_argument("--index-dir", default="data/processed/indexes")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = check_data_leak(
        eval_file=Path(args.eval_file),
        index_dir=Path(args.index_dir),
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {out}")

    sys.exit(0 if result["safe"] else 1)


if __name__ == "__main__":
    main()
