from __future__ import annotations

from medical_rag.schema import FusedEvidence
from medical_rag.text_utils import lexical_overlap


class LightweightReranker:
    """Simple reranker that rewards query-evidence lexical overlap."""

    def rerank(self, query: str, evidence: list[FusedEvidence]) -> list[FusedEvidence]:
        rescored: list[FusedEvidence] = []
        for item in evidence:
            overlap = lexical_overlap(query, item.text)
            copy = item.model_copy(deep=True)
            copy.component_scores["lexical_overlap"] = overlap
            copy.fused_score = copy.fused_score + 0.05 * overlap
            rescored.append(copy)
        return sorted(rescored, key=lambda item: item.fused_score, reverse=True)
