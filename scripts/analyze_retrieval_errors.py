"""Analyze retrieval/evaluation errors from advanced benchmark JSON.

This script is intentionally offline and API-free. It reads an existing
`evaluate-advanced` JSON output and writes a compact Markdown error analysis
for thesis/debugging.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    if value is None:
        return "-"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def build_error_report(benchmark: dict[str, Any], max_examples: int = 20) -> str:
    rows = list(benchmark.get("rows", []))
    total = len(rows)
    misses = [row for row in rows if not row.get("hit")]
    route_misses = [row for row in rows if row.get("route_ok") is False]

    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    error_counts: Counter[str] = Counter()
    for row in rows:
        dataset = row.get("dataset_hint") or row.get("dataset") or "unknown"
        if row.get("hit"):
            by_dataset[dataset]["hit"] += 1
        else:
            by_dataset[dataset]["miss"] += 1
        for category in row.get("error_categories", []) or []:
            error_counts[str(category)] += 1

    dataset_rows = []
    for dataset, counts in sorted(by_dataset.items()):
        ds_total = counts["hit"] + counts["miss"]
        dataset_rows.append([
            dataset,
            ds_total,
            counts["hit"],
            counts["miss"],
            counts["hit"] / ds_total if ds_total else 0.0,
        ])

    miss_rows = []
    for row in misses[:max_examples]:
        miss_rows.append([
            row.get("dataset_hint") or row.get("dataset") or "unknown",
            str(row.get("gold") or "-")[:48],
            str(row.get("query") or "")[:80],
            ", ".join(str(eid) for eid in row.get("evidence_ids", [])[:3]),
        ])

    content = [
        "# Retrieval Error Analysis",
        "",
        "## Summary",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Total rows", total],
                ["Retrieval misses", len(misses)],
                ["Route misses", len(route_misses)],
                ["Recall proxy", (total - len(misses)) / total if total else 0.0],
            ],
        ),
        "## Per-dataset Retrieval",
        "",
        _table(["Dataset", "Cases", "Hits", "Misses", "Hit rate"], dataset_rows) if dataset_rows else "_No per-dataset rows._\n",
        "## Error Categories",
        "",
        _table(["Category", "Count"], [[key, value] for key, value in error_counts.most_common()]) if error_counts else "_No error categories recorded._\n",
        "## Top Miss Examples",
        "",
        _table(["Dataset", "Gold ID", "Query", "Top evidence IDs"], miss_rows) if miss_rows else "_No retrieval misses._\n",
    ]
    return "\n".join(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Markdown retrieval error analysis from benchmark JSON")
    parser.add_argument("--benchmark-file", default="outputs/benchmark/baseline_advanced.json")
    parser.add_argument("--output-file", default="outputs/benchmark/retrieval_error_analysis.md")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    report = build_error_report(_load_json(Path(args.benchmark_file)), max_examples=args.max_examples)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
