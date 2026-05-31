from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_rag.config import RAGConfig
from medical_rag.evaluation import evaluate
from medical_rag.pipeline import MedicalRAGPipeline


def profile_config(name: str, base: RAGConfig) -> RAGConfig:
    cfg = base.model_copy(deep=True)
    cfg.profile_name = name
    if name == "A":
        cfg.force_text_only = True
        cfg.enable_rerank = False
        cfg.image_weight = 0.0
    elif name == "B":
        cfg.force_text_only = True
        cfg.enable_rerank = True
        cfg.image_weight = 0.0
    elif name == "C":
        cfg.force_image_for_image_queries = True
        cfg.enable_rerank = False
        cfg.bm25_weight = 0.35
        cfg.dense_weight = 0.25
        cfg.image_weight = 0.70
    elif name == "D":
        cfg.force_text_only = False
        cfg.force_image_for_image_queries = True
        cfg.enable_rerank = True
        cfg.bm25_weight = 0.55
        cfg.dense_weight = 0.45
        cfg.image_weight = 0.50
    else:
        raise ValueError(f"Unknown ablation profile: {name}")
    return cfg


def run_ablation(
    eval_file: Path,
    index_dir: Path,
    data_dir: Path,
    output_dir: Path,
    top_k: int = 5,
    profiles: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = RAGConfig(index_dir=index_dir, data_dir=data_dir).resolved()
    profiles = profiles or ["A", "B", "C", "D"]
    results: dict[str, Any] = {}

    for profile in profiles:
        cfg = profile_config(profile, base)
        pipeline = MedicalRAGPipeline(cfg)
        metrics = evaluate(eval_file, pipeline, top_k=top_k)
        metrics["profile"] = profile
        results[profile] = metrics
        (output_dir / f"config_{profile.lower()}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "profiles": profiles,
        "ranking": sorted(
            [
                {
                    "profile": profile,
                    "recall_at_k": metrics.get("recall_at_k", 0.0),
                    "mrr_at_k": metrics.get("mrr_at_k", 0.0),
                    "routing_accuracy": metrics.get("routing_accuracy", 0.0),
                    "image_recall_at_k": metrics.get("per_modality", {}).get("image", {}).get("recall_at_k", 0.0),
                }
                for profile, metrics in results.items()
            ],
            key=lambda item: (item["recall_at_k"], item["mrr_at_k"]),
            reverse=True,
        ),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
