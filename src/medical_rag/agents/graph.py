"""LangGraph StateGraph assembly for Medical Multimodal RAG Agent.

Graph flow (from project_description.md):

    decompose_query
         │
         ├─ (text)   ──→ retrieve_text  ──────────────→ fuse_and_verify
         │                                                      │
         ├─ (visual) ──→ retrieve_visual ─────────────→ fuse_and_verify
         │                                                      │
         └─ (hybrid) ──→ retrieve_text + retrieve_visual ──────┘
                                                                │
                                                       generate_answer → END

Reference: HM-RAG (src/support_repo/HMRAG_project/HMRAG/agents/) for
LangGraph node structure and conditional routing patterns.
"""
from __future__ import annotations

import logging
from typing import Any

from medical_rag.agents.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_decompose(state: AgentState) -> str:
    """Conditional router after decompose_query node."""
    query_type = state.get("query_type", "text")
    if query_type == "visual":
        return "visual_only"
    if query_type == "hybrid":
        return "hybrid"
    return "text_only"


def build_agent_graph(config: dict[str, Any] | None = None) -> Any:
    """Build and compile the LangGraph StateGraph.

    Args:
        config: optional config dict injected into state at START

    Returns:
        Compiled LangGraph graph ready for .invoke()

    Raises:
        RuntimeError: if langgraph is not installed
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph not installed. Install: pip install langgraph>=0.2.28"
        ) from exc

    from medical_rag.agents.decision import fuse_and_verify, generate_answer
    from medical_rag.agents.decomposer import decompose_query
    from medical_rag.agents.text_retriever import retrieve_text
    from medical_rag.agents.visual_retriever import retrieve_visual

    cfg = config or {}

    # Wrap nodes to inject config into state
    def _decompose(state: AgentState) -> AgentState:
        return decompose_query({**state, "config": cfg})

    def _retrieve_text(state: AgentState) -> AgentState:
        return retrieve_text({**state, "config": cfg})

    def _retrieve_visual(state: AgentState) -> AgentState:
        return retrieve_visual({**state, "config": cfg})

    def _fuse(state: AgentState) -> AgentState:
        return fuse_and_verify({**state, "config": cfg})

    def _generate(state: AgentState) -> AgentState:
        return generate_answer({**state, "config": cfg})

    # Text-only path: retrieve_text → fuse → generate
    def _retrieve_text_then_fuse(state: AgentState) -> AgentState:
        s = _retrieve_text(state)
        s = {**state, **s}
        return s

    # Visual-only path: retrieve_visual → fuse → generate
    def _retrieve_visual_then_fuse(state: AgentState) -> AgentState:
        s = _retrieve_visual(state)
        s = {**state, **s}
        return s

    # Hybrid path: both retrievers → fuse → generate
    def _retrieve_hybrid(state: AgentState) -> AgentState:
        text_state = _retrieve_text(state)
        visual_state = _retrieve_visual(state)
        return {
            **state,
            **text_state,
            **visual_state,
        }

    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("decompose", _decompose)
    builder.add_node("text_only", _retrieve_text_then_fuse)
    builder.add_node("visual_only", _retrieve_visual_then_fuse)
    builder.add_node("hybrid", _retrieve_hybrid)
    builder.add_node("fuse", _fuse)
    builder.add_node("generate", _generate)

    # Edges
    builder.set_entry_point("decompose")
    builder.add_conditional_edges(
        "decompose",
        _route_after_decompose,
        {
            "text_only": "text_only",
            "visual_only": "visual_only",
            "hybrid": "hybrid",
        },
    )
    builder.add_edge("text_only", "fuse")
    builder.add_edge("visual_only", "fuse")
    builder.add_edge("hybrid", "fuse")
    builder.add_edge("fuse", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


class AgenticRAGPipeline:
    """High-level pipeline using the LangGraph agent graph.

    Drop-in replacement for MedicalRAGPipeline that uses agents.
    Falls back to the baseline pipeline if LangGraph not installed.
    """

    def __init__(self, config_dict: dict[str, Any]):
        self.config_dict = config_dict
        self._graph: Any = None

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self._graph = build_agent_graph(self.config_dict)
        return self._graph

    def run(
        self,
        question: str,
        image_path: str | None = None,
        dataset_hint: str | None = None,
    ) -> dict[str, Any]:
        """Run the agent pipeline.

        Args:
            question: medical query
            image_path: optional uploaded image path
            dataset_hint: optional dataset name to restrict retrieval

        Returns:
            State dict with answer, citations, reasoning_steps, etc.
        """
        initial_state: AgentState = {
            "question": question,
            "image_path": image_path,
            "dataset_hint": dataset_hint,
            "step_count": 0,
            "reasoning_steps": [],
        }
        try:
            result = self.graph.invoke(initial_state)
            return dict(result)
        except Exception as exc:
            logger.error(f"Agent pipeline failed: {exc}")
            return {
                "question": question,
                "answer": f"Agent pipeline error: {exc}",
                "citations": [],
                "reasoning_steps": [f"error: {exc}"],
                "error": str(exc),
            }
