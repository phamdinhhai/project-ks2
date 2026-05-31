from __future__ import annotations

from collections import defaultdict

from medical_rag.config import RAGConfig
from medical_rag.schema import FusedEvidence, Modality, RetrievalResult


class LateFusion:
    """Rank-based late fusion using weighted reciprocal rank fusion."""

    def __init__(self, config: RAGConfig):
        self.config = config

    def _to_fused(
        self,
        result: RetrievalResult,
        score: float,
        component_scores: dict[str, float],
    ) -> FusedEvidence:
        return FusedEvidence(
            id=result.id,
            modality=result.modality,
            fused_score=float(score),
            text=result.text,
            dataset=result.dataset,
            source_path=result.source_path,
            component_scores=component_scores,
            metadata=result.metadata,
        )

    def fuse(
        self,
        text_results: list[RetrievalResult],
        image_results: list[RetrievalResult],
        prefer_image: bool = False,
    ) -> list[FusedEvidence]:
        scores: dict[str, float] = defaultdict(float)
        components: dict[str, dict[str, float]] = defaultdict(dict)
        payload: dict[str, RetrievalResult] = {}

        for result in text_results:
            contribution = 1.0 / (self.config.rrf_k + result.rank)
            scores[result.id] += contribution
            components[result.id]["text_rrf"] = contribution
            components[result.id]["text_score"] = result.score
            payload[result.id] = result

        for result in image_results:
            contribution = self.config.image_weight / (self.config.rrf_k + result.rank)
            scores[result.id] += contribution
            components[result.id]["image_rrf"] = contribution
            components[result.id]["image_score"] = result.score
            payload[result.id] = result

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        fused: list[FusedEvidence] = [
            self._to_fused(payload[result_id], score, components[result_id])
            for result_id, score in ordered[: self.config.final_top_k]
        ]

        if prefer_image and image_results:
            has_image = any(item.modality == Modality.IMAGE for item in fused)
            if not has_image:
                top_image = image_results[0]
                image_score = scores.get(top_image.id, 0.0)
                image_fused = self._to_fused(top_image, image_score, components.get(top_image.id, {}))
                if fused:
                    fused[-1] = image_fused
                else:
                    fused = [image_fused]
                fused = sorted(fused, key=lambda item: item.fused_score, reverse=True)[: self.config.final_top_k]
        return fused
