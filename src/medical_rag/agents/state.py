"""Agent state definition for the LangGraph medical RAG pipeline.

Reference: HM-RAG (src/support_repo/HMRAG_project) uses a TypedDict-based
state that flows through decomposition → retrieval → decision → generation.
A-MAR (src/support_repo/A-MAR_project) adds adaptive loop state with step_count.
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    All fields are optional (total=False) so each node only updates
    the fields it produces. LangGraph merges state across nodes.
    """

    # --- Input ---
    question: str
    image_path: str | None

    # --- Decomposition output ---
    text_subquery: str
    visual_subquery: str
    query_type: str  # "text" | "visual" | "hybrid"

    # --- Retrieval results ---
    text_evidence: list[dict[str, Any]]    # [{text, score, source, id}, ...]
    visual_evidence: list[dict[str, Any]]  # [{image_path, patch_id, caption, score, bbox}, ...]

    # --- Decision output ---
    fused_evidence: str     # formatted evidence context string
    faithfulness_score: float

    # --- Final output ---
    answer: str
    reasoning_steps: list[str]
    citations: list[str]

    # --- Agent metadata ---
    step_count: int
    error: str | None
    dataset_hint: str | None
    config: dict[str, Any]
