from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_rag.config import RAGConfig
from medical_rag.evaluation import evaluate
from medical_rag.evaluation_advanced import evaluate_agent
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


def agent_profile_config(
    name: str,
    index_dir: Path,
    data_dir: Path,
    top_k: int = 5,
    use_mock_models: bool = True,
) -> dict[str, Any]:
    """Return config dict for AgenticRAGPipeline ablation profiles E/F/G/H."""
    import os

    base: dict[str, Any] = {
        "index_dir": str(index_dir),
        "data_dir": str(data_dir),
        "use_qdrant": False,
        "use_vlm_generation": False,
        "use_fine_grained_visual": False,
        "use_cross_encoder_rerank": False,
        "text_top_k": top_k,
        "image_top_k": top_k,
        "qdrant_url": os.environ.get("QDRANT_URL", ":memory:"),
        "qdrant_api_key": os.environ.get("QDRANT_API_KEY"),
        "use_cloud_auth": bool(os.environ.get("QDRANT_API_KEY")),
        "llm_provider": "mock" if use_mock_models else ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else "auto"),
        "openrouter_model": os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        "openrouter_base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "use_mock_models": use_mock_models,
        "agent_profile": name,
    }

    if name == "E":
        base.update({
            "force_text_only_agent": True,
            "use_vlm_generation": False,
            "use_fine_grained_visual": False,
        })
    elif name == "F":
        base.update({
            "force_text_only_agent": False,
            "use_vlm_generation": False,
            "use_fine_grained_visual": False,
        })
    elif name == "G":
        base.update({
            "force_text_only_agent": False,
            "use_vlm_generation": True,
            "use_fine_grained_visual": False,
        })
    elif name == "H":
        base.update({
            "force_text_only_agent": False,
            "use_vlm_generation": True,
            "use_fine_grained_visual": True,
        })
    else:
        raise ValueError(f"Unknown agent ablation profile: {name}")
    return base


def run_agent_ablation(
    eval_file: Path,
    index_dir: Path,
    data_dir: Path,
    output_dir: Path,
    top_k: int = 5,
    profiles: list[str] | None = None,
    use_mock_models: bool = True,
) -> dict[str, Any]:
    """Run AgenticRAGPipeline ablation profiles E/F/G/H."""
    from medical_rag.agents.graph import AgenticRAGPipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = profiles or ["E", "F", "G", "H"]
    results: dict[str, Any] = {}

    for profile in profiles:
        cfg = agent_profile_config(
            profile,
            index_dir=index_dir,
            data_dir=data_dir,
            top_k=top_k,
            use_mock_models=use_mock_models,
        )
        agent = AgenticRAGPipeline(cfg)
        metrics = evaluate_agent(eval_file, agent, top_k=top_k)
        metrics["profile"] = profile
        metrics["config"] = cfg
        results[profile] = metrics
        (output_dir / f"agent_config_{profile.lower()}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "profiles": profiles,
        "ranking": sorted(
            [
                {
                    "profile": profile,
                    "exact_match": metrics.get("exact_match", 0.0),
                    "token_f1": metrics.get("token_f1", 0.0),
                    "mean_steps": metrics.get("mean_steps", 0.0),
                    "error_count": metrics.get("error_count", 0),
                }
                for profile, metrics in results.items()
            ],
            key=lambda item: (item["token_f1"], -item["error_count"]),
            reverse=True,
        ),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
