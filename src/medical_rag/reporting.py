from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_rag.pipeline import MedicalRAGPipeline
from medical_rag.schema import FusedEvidence, GeneratedAnswer


def evidence_to_dict(item: FusedEvidence, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": item.id,
        "dataset": item.dataset,
        "modality": item.modality.value,
        "fused_score": item.fused_score,
        "source_path": item.source_path,
        "component_scores": item.component_scores,
        "metadata": item.metadata,
        "text_preview": " ".join(item.text.split())[:500],
    }


def answer_to_debug_dict(answer: GeneratedAnswer) -> dict[str, Any]:
    return {
        "query": answer.intent.query,
        "intent": answer.intent.model_dump(mode="json"),
        "answer": answer.answer,
        "citations": answer.citations,
        "evidence": [evidence_to_dict(item, rank) for rank, item in enumerate(answer.evidence, start=1)],
    }


def run_query_debug(
    pipeline: MedicalRAGPipeline,
    query: str,
    image_path: str | None = None,
    dataset_hint: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    answer = pipeline.run(query, image_path=image_path, dataset_hint=dataset_hint, top_k=top_k)
    return answer_to_debug_dict(answer)


def write_query_debug(payload: dict[str, Any], output_file: Path) -> None:
    import json

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
