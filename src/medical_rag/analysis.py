from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_ablation(ablation_dir: Path, output_file: Path | None = None) -> str:
    summary_path = ablation_dir / "summary.json"
    if not summary_path.exists():
        return f"Ablation summary not found at {summary_path}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary.get("results", {})

    lines = [
        "# Medical Multimodal RAG — Ablation Study Report",
        "",
        "## Summary Table",
        "",
        "| Profile | Description | Recall@5 | MRR@5 | Ans Acc | Routing | Image Recall |",
        "|---|---|---|---|---|---|---|",
    ]

    profile_descs = {
        "A": "Text-only baseline (TF-IDF)",
        "B": "Text-only + Rerank",
        "C": "Image branch enabled (Weighted RRF)",
        "D": "Full pipeline: Text + Image + Fusion + Rerank",
    }

    for profile in ["A", "B", "C", "D"]:
        metrics = results.get(profile)
        if not metrics:
            continue
        desc = profile_descs.get(profile, "Unknown")
        recall = metrics.get("recall_at_k", 0.0)
        mrr = metrics.get("mrr_at_k", 0.0)
        ans_acc = metrics.get("answer_accuracy", 0.0)
        route = metrics.get("routing_accuracy", 0.0)
        img_recall = metrics.get("per_modality", {}).get("image", {}).get("recall_at_k", 0.0)

        lines.append(f"| **{profile}** | {desc} | {recall:.4f} | {mrr:.4f} | {ans_acc:.4f} | {route:.4f} | {img_recall:.4f} |")

    lines.append("")
    lines.append("## Per-Dataset Recall@5")
    lines.append("")

    datasets = sorted({ds for m in results.values() for ds in m.get("dataset_distribution", {})})
    if datasets:
        header = "| Profile | " + " | ".join(datasets) + " |"
        sep = "|---| " + " | ".join(["---"] * len(datasets)) + " |"
        lines.append(header)
        lines.append(sep)

        for profile in ["A", "B", "C", "D"]:
            metrics = results.get(profile)
            if not metrics:
                continue
            # Note: current metrics only has global distribution, not per-dataset recall yet in a direct way.
            # We'll use global recall for now in the summary or skip if not computed per dataset.
            row = f"| **{profile}** | " + " | ".join([f"{metrics.get('recall_at_k', 0.0):.4f}"] * len(datasets)) + " |"
            # lines.append(row) # Skip for now as it needs more granular calculation in evaluation.py

    lines.append("")
    lines.append("## Failure Analysis (Full System - Config D)")
    lines.append("")

    config_d = results.get("D")
    if config_d:
        errors = config_d.get("error_distribution", {})
        lines.append("| Error Category | Count | Percentage |")
        lines.append("|---|---|---|")
        total = config_d.get("total", 1)
        for cat, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            if cat == "ok":
                continue
            lines.append(f"| {cat} | {count} | {(count/total)*100:.1f}% |")

    lines.append("")
    lines.append("---")
    lines.append(f"Report generated for {ablation_dir.name}")

    report = "\n".join(lines)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report, encoding="utf-8")

    return report
