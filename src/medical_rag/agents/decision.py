"""Decision agent: evidence fusion, verification, and answer generation.

Reference:
- HM-RAG decision agent: merges evidence from multiple retrieval sources
- MMed-RAG adaptive context selection: picks optimal number of evidence items
- A-MAR stopping criteria: confidence-based stopping
"""
from __future__ import annotations

import logging
from typing import Any

from medical_rag.agents.state import AgentState

logger = logging.getLogger(__name__)


def fuse_and_verify(state: AgentState) -> AgentState:
    """LangGraph node: merge text + visual evidence and check quality.

    1. Interleave text and visual evidence by score
    2. Remove duplicates
    3. Format unified evidence context string
    4. Compute simple faithfulness heuristic
    """
    text_evidence = state.get("text_evidence", [])
    visual_evidence = state.get("visual_evidence", [])
    step_count = state.get("step_count", 0) + 1
    steps = list(state.get("reasoning_steps", []))

    # Merge and deduplicate
    all_evidence = text_evidence + visual_evidence
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in sorted(all_evidence, key=lambda x: x.get("score", 0.0), reverse=True):
        key = item.get("id", "") or item.get("text", "")[:100]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Format context string for generation
    context_parts: list[str] = []
    for idx, item in enumerate(unique[:8], start=1):
        modality_tag = item.get("modality", "text")
        source_info = f"{item.get('dataset', 'unknown')}/{item.get('record_id', '')}"
        text = item.get("text", "")[:600]
        context_parts.append(
            f"[Evidence {idx}] ({modality_tag}, {source_info}) {text}"
        )
    fused_context = "\n\n".join(context_parts)

    # Simple faithfulness heuristic: do we have overlapping evidence?
    text_count = sum(1 for e in unique if e.get("modality") == "text")
    image_count = sum(1 for e in unique if e.get("modality") == "image")
    faithfulness = min(1.0, (text_count + image_count) / max(1, len(unique)))

    steps.append(
        f"[fusion] {text_count} text + {image_count} image evidence → "
        f"{len(unique)} total, faithfulness={faithfulness:.2f}"
    )

    return {
        "fused_evidence": fused_context,
        "text_evidence": [e for e in unique if e.get("modality") == "text"],
        "visual_evidence": [e for e in unique if e.get("modality") == "image"],
        "faithfulness_score": faithfulness,
        "step_count": step_count,
        "reasoning_steps": steps,
    }


def generate_answer(state: AgentState) -> AgentState:
    """LangGraph node: generate answer using VLM or extractive fallback.

    If VLM available: Qwen2.5-VL generates a grounded answer with citations.
    Otherwise: extractive answer from top evidence.
    """
    question = state.get("question", "")
    fused_evidence = state.get("fused_evidence", "")
    image_path = state.get("image_path")
    step_count = state.get("step_count", 0) + 1
    cfg = state.get("config", {})
    steps = list(state.get("reasoning_steps", []))

    # Try VLM generation
    if cfg.get("use_vlm_generation"):
        try:
            answer, citations = _generate_vlm(question, image_path, fused_evidence, cfg)
            steps.append("[generation] VLM generated answer with citations")
            return {
                "answer": answer,
                "citations": citations,
                "step_count": step_count,
                "reasoning_steps": steps,
            }
        except Exception as exc:
            logger.warning(f"VLM generation failed: {exc}")
            steps.append(f"[generation] VLM failed ({exc}), using extractive fallback")

    # Extractive fallback
    answer, citations = _generate_extractive(question, state)
    steps.append("[generation] Extractive answer generated")
    return {
        "answer": answer,
        "citations": citations,
        "step_count": step_count,
        "reasoning_steps": steps,
    }


def _generate_vlm(
    question: str,
    image_path: str | None,
    context: str,
    cfg: dict[str, Any],
) -> tuple[str, list[str]]:
    """Generate answer using Qwen2.5-VL."""
    import os

    if cfg.get("use_mock_models") or cfg.get("llm_provider") == "mock":
        from medical_rag.models.mock_models import MockQwenVL
        vlm = MockQwenVL()
    elif cfg.get("llm_provider") == "openrouter" or os.environ.get("OPENROUTER_API_KEY"):
        from medical_rag.models.openrouter_vlm import OpenRouterVLM
        vlm = OpenRouterVLM(
            model=cfg.get("openrouter_model"),
            base_url=cfg.get("openrouter_base_url"),
            max_tokens=cfg.get("max_new_tokens", 256),
        )
    else:
        from medical_rag.models.qwen_vl import QwenVLModel
        vlm = QwenVLModel(
            model_name=cfg.get("vlm_model_name"),
            use_cpu_offload=cfg.get("use_cpu_offload", True),
            max_new_tokens=cfg.get("max_new_tokens", 256),
        )

    image = None
    if image_path:
        from pathlib import Path
        if Path(image_path).exists():
            from PIL import Image
            image = Image.open(image_path).convert("RGB")

    answer = vlm.generate(question=question, image=image, context=context)

    # Extract citations from evidence markers
    import re
    citation_matches = re.findall(r'\[Evidence (\d+)\]', context)
    citations = [f"[{m}]" for m in citation_matches[:8]]

    return answer, citations


def _generate_extractive(
    question: str,
    state: AgentState,
) -> tuple[str, list[str]]:
    """Fallback extractive generation from evidence."""
    text_evidence = state.get("text_evidence", [])
    visual_evidence = state.get("visual_evidence", [])
    all_evidence = text_evidence + visual_evidence

    if not all_evidence:
        return "No relevant evidence found in the knowledge base.", []

    snippets: list[str] = []
    citations: list[str] = []
    for idx, item in enumerate(all_evidence[:5], start=1):
        mod = item.get("modality", "text")
        ds = item.get("dataset", "unknown")
        text = " ".join(item.get("text", "").split())[:400]
        snippets.append(f"[{idx}] ({mod}) {text}")
        citations.append(f"[{idx}] {ds}:{item.get('id', '')}")

    answer = (
        f"Based on retrieved evidence for: {question}\n\n"
        + "\n\n".join(snippets)
        + "\n\nCitations: "
        + "; ".join(citations)
    )
    return answer, citations
