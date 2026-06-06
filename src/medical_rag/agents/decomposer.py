"""Query decomposition agent node.

Decomposes a medical query into text_subquery + visual_subquery and routes to
text-only / visual-only / hybrid retrieval.

Reference:
- HM-RAG decomposer: creates atomic sub-queries from complex queries
- A-MAR: structured reasoning plan before decomposition
"""
from __future__ import annotations

import json
import logging
import re

from medical_rag.agents.state import AgentState
from medical_rag.router import QueryRouter

logger = logging.getLogger(__name__)

# Rule-based router for when VLM is not available
_ROUTER = QueryRouter()

_DECOMPOSE_PROMPT = """\
You are a medical query analyzer. Given the question below, decompose it into:
1. text_subquery: what to search in medical text documents (clinical knowledge)
2. visual_subquery: what to search in medical images (radiology/pathology)
3. query_type: "text" (no image needed), "visual" (only image relevant), or "hybrid" (both needed)

Question: {question}

Return ONLY a valid JSON object:
{{"text_subquery": "...", "visual_subquery": "...", "query_type": "text|visual|hybrid"}}
"""


def decompose_query(state: AgentState) -> AgentState:
    """LangGraph node: decompose query into sub-queries and determine routing.

    Uses VLM if available, falls back to rule-based router.

    Args:
        state: current agent state with 'question' (and optionally 'image_path')

    Returns:
        Updated state with text_subquery, visual_subquery, query_type, step_count
    """
    question = state.get("question", "")
    image_path = state.get("image_path")
    step_count = state.get("step_count", 0) + 1
    cfg = state.get("config", {})

    updates: AgentState = {"step_count": step_count}

    if cfg.get("force_text_only_agent"):
        updates["text_subquery"] = question
        updates["visual_subquery"] = ""
        updates["query_type"] = "text"
        updates["reasoning_steps"] = state.get("reasoning_steps", []) + [
            "[decompose] Forced text-only agent profile"
        ]
        return updates

    # Try VLM-based decomposition
    vlm = _try_get_vlm(state)
    if vlm is not None:
        try:
            prompt = _DECOMPOSE_PROMPT.format(question=question)
            raw = vlm.generate(prompt)
            parsed = _parse_json_response(raw)
            if parsed:
                updates["text_subquery"] = str(parsed.get("text_subquery", question))
                updates["visual_subquery"] = str(parsed.get("visual_subquery", question))
                updates["query_type"] = _validate_query_type(
                    parsed.get("query_type", "text"), image_path
                )
                updates["reasoning_steps"] = state.get("reasoning_steps", []) + [
                    f"[decompose] VLM decomposed query → type={updates['query_type']}"
                ]
                logger.info(f"VLM decomposition: type={updates['query_type']}")
                return updates
        except Exception as exc:
            logger.warning(f"VLM decomposition failed, using rule-based: {exc}")

    # Rule-based fallback
    intent = _ROUTER.route(question, image_path=image_path)
    if intent.use_image_branch:
        query_type = "hybrid" if question.strip() else "visual"
    else:
        query_type = "text"

    updates["text_subquery"] = question
    updates["visual_subquery"] = question
    updates["query_type"] = query_type
    updates["reasoning_steps"] = state.get("reasoning_steps", []) + [
        f"[decompose] Rule-based router → type={query_type}, lang={intent.language.value}"
    ]
    logger.info(f"Rule-based decomposition: type={query_type}")
    return updates


def _try_get_vlm(state: AgentState) -> object | None:
    """Try to get VLM from state config, return None if not configured."""
    cfg = state.get("config", {})
    if not cfg.get("use_vlm_generation"):
        return None
    try:
        if cfg.get("use_mock_models") or cfg.get("llm_provider") == "mock":
            from medical_rag.models.mock_models import MockQwenVL
            return MockQwenVL()
        if cfg.get("llm_provider") == "openrouter":
            from medical_rag.models.openrouter_vlm import OpenRouterVLM
            return OpenRouterVLM(
                model=cfg.get("openrouter_model"),
                base_url=cfg.get("openrouter_base_url"),
                max_tokens=cfg.get("max_new_tokens", 256),
            )
        from medical_rag.models.qwen_vl import QwenVLModel
        return QwenVLModel(model_name=cfg.get("vlm_model_name", "Qwen/Qwen2.5-VL-7B-Instruct"))
    except Exception:
        return None


def _parse_json_response(raw: str) -> dict | None:
    """Extract JSON from a potentially mixed text response."""
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _validate_query_type(query_type: str, image_path: str | None) -> str:
    """Ensure query_type is valid and consistent with image availability."""
    valid = {"text", "visual", "hybrid"}
    qtype = query_type.lower() if query_type else "text"
    if qtype not in valid:
        qtype = "text"
    # If no image supplied, can't do visual-only
    if qtype == "visual" and not image_path:
        qtype = "text"
    return qtype
