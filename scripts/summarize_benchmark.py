"""Summarize benchmark outputs into a thesis-friendly Markdown report.

Usage:
    python scripts/summarize_benchmark.py
    python scripts/summarize_benchmark.py --benchmark-file outputs/benchmark/baseline_advanced.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    if value is None:
        return "-"
    return str(value)


def _ablation_table(summary: dict[str, Any]) -> str:
    results = summary.get("results", {})
    if not results:
        return "_No ablation results found._\n"

    lines = [
        "| Profile | Recall@5 | MRR@5 | Answer Acc | Routing | Image Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile in sorted(results):
        metrics = results[profile]
        image_recall = metrics.get("per_modality", {}).get("image", {}).get("recall_at_k")
        lines.append(
            "| {profile} | {recall} | {mrr} | {ans} | {routing} | {image} |".format(
                profile=profile,
                recall=_fmt(metrics.get("recall_at_k")),
                mrr=_fmt(metrics.get("mrr_at_k")),
                ans=_fmt(metrics.get("answer_accuracy")),
                routing=_fmt(metrics.get("routing_accuracy")),
                image=_fmt(image_recall),
            )
        )
    return "\n".join(lines) + "\n"


def _advanced_overall_table(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "_No advanced benchmark found._\n"
    rows = [
        ("Total cases", metrics.get("total")),
        ("Recall@5", metrics.get("recall_at_k")),
        ("MRR@5", metrics.get("mrr_at_k")),
        ("Routing accuracy", metrics.get("routing_accuracy")),
        ("Exact Match", metrics.get("exact_match")),
        ("Token F1", metrics.get("token_f1")),
    ]
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {name} | {_fmt(value)} |" for name, value in rows)
    return "\n".join(lines) + "\n"


def _per_dataset_table(metrics: dict[str, Any]) -> str:
    per_dataset = metrics.get("per_dataset", {})
    if not per_dataset:
        return "_No per-dataset metrics found._\n"
    lines = [
        "| Dataset | Cases | Recall@5 | MRR@5 | EM | Token F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(per_dataset):
        row = per_dataset[dataset]
        lines.append(
            "| {dataset} | {total} | {recall} | {mrr} | {em} | {f1} |".format(
                dataset=dataset,
                total=_fmt(row.get("total")),
                recall=_fmt(row.get("recall_at_k")),
                mrr=_fmt(row.get("mrr_at_k")),
                em=_fmt(row.get("exact_match")),
                f1=_fmt(row.get("token_f1")),
            )
        )
    return "\n".join(lines) + "\n"


def _error_table(metrics: dict[str, Any]) -> str:
    errors = metrics.get("error_distribution", {})
    total = int(metrics.get("total") or 0)
    if not errors:
        return "_No error distribution found._\n"
    lines = ["| Error Category | Count | Rate |", "|---|---:|---:|"]
    for key, count in sorted(errors.items(), key=lambda item: item[1], reverse=True):
        rate = (float(count) / float(total)) if total else 0.0
        lines.append(f"| {key} | {count} | {_fmt(rate)} |")
    return "\n".join(lines) + "\n"


def _agent_ablation_table(summary: dict[str, Any]) -> str:
    results = summary.get("results", {})
    ranking = summary.get("ranking", [])
    if not results and not ranking:
        return "_No agent ablation results found._\n"

    rows = ranking or [
        {
            "profile": profile,
            "exact_match": metrics.get("exact_match", 0.0),
            "token_f1": metrics.get("token_f1", 0.0),
            "mean_steps": metrics.get("mean_steps", 0.0),
            "error_count": metrics.get("error_count", 0),
        }
        for profile, metrics in sorted(results.items())
    ]
    lines = [
        "| Profile | EM | Token F1 | Mean Steps | Error Count |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {em} | {f1} | {steps} | {errors} |".format(
                profile=row.get("profile", "-"),
                em=_fmt(row.get("exact_match")),
                f1=_fmt(row.get("token_f1")),
                steps=_fmt(row.get("mean_steps")),
                errors=row.get("error_count", "-"),
            )
        )
    return "\n".join(lines) + "\n"


def _ragas_table(summary: dict[str, Any]) -> str:
    if not summary:
        return "_No RAGAS results found._\n"
    if summary.get("skipped"):
        reason = summary.get("reason", "RAGAS was skipped.")
        return f"_RAGAS skipped_: {reason}\n"
    metrics = summary.get("metrics", {})
    if not metrics:
        return "_No RAGAS metric values found._\n"
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.append(f"| Evaluated rows | {summary.get('evaluated_rows', 0)} |")
    for name, value in sorted(metrics.items()):
        lines.append(f"| {name} | {_fmt(value)} |")
    return "\n".join(lines) + "\n"


def build_report(
    ablation_summary: Path,
    benchmark_file: Path,
    output_file: Path,
    agent_summary: Path | None = None,
    ragas_file: Path | None = None,
) -> str:
    ablation = _load_json(ablation_summary)
    advanced = _load_json(benchmark_file)
    agent = _load_json(agent_summary) if agent_summary else {}
    ragas = _load_json(ragas_file) if ragas_file else {}

    content = f"""# Benchmark Summary

Generated from:
- `{ablation_summary}`
- `{benchmark_file}`

## Ablation A/B/C/D

{_ablation_table(ablation)}
## Advanced Baseline Overall

{_advanced_overall_table(advanced)}
## Advanced Baseline Per Dataset

{_per_dataset_table(advanced)}
## Error Distribution

{_error_table(advanced)}
## Agent Ablation E/F/G/H

{_agent_ablation_table(agent)}
## RAGAS Smoke Metrics

{_ragas_table(ragas)}
## Notes for Thesis

- Config C currently has the highest local lexical-baseline Recall@5.
- Config D has slightly lower recall after reranking but remains the intended full baseline.
- EM/F1 are low because the current baseline generator outputs evidence drafts rather than short-form answers.
- ROCO and MIMIC-CXR zero-recall issue was fixed by aligning eval cases with canonical dataset manifests.
"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark outputs into Markdown")
    parser.add_argument("--ablation-summary", default="outputs/ablation_full/summary.json")
    parser.add_argument("--benchmark-file", default="outputs/benchmark/baseline_advanced.json")
    parser.add_argument("--agent-summary", default="outputs/agent_ablation_smoke/summary.json")
    parser.add_argument("--ragas-file", default="outputs/benchmark/ragas_smoke.json")
    parser.add_argument("--output-file", default="outputs/benchmark/benchmark_summary.md")
    args = parser.parse_args()

    report = build_report(
        ablation_summary=Path(args.ablation_summary),
        benchmark_file=Path(args.benchmark_file),
        agent_summary=Path(args.agent_summary) if args.agent_summary else None,
        ragas_file=Path(args.ragas_file) if args.ragas_file else None,
        output_file=Path(args.output_file),
    )
    print(report)
    print(f"\nSaved to {args.output_file}")


if __name__ == "__main__":
    main()
