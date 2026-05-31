from __future__ import annotations

from pathlib import Path

from medical_rag.config import RAGConfig
from medical_rag.generation import ExtractiveGenerator
from medical_rag.indexing import load_indexes
from medical_rag.retrieval.fusion import LateFusion
from medical_rag.retrieval.image import ImageRetriever
from medical_rag.retrieval.rerank import LightweightReranker
from medical_rag.retrieval.text import HybridTextRetriever
from medical_rag.router import QueryRouter
from medical_rag.schema import GeneratedAnswer, Modality


class MedicalRAGPipeline:
    """End-to-end text-first, conditional-image, late-fusion RAG pipeline."""

    def __init__(self, config: RAGConfig, index_bundle: dict | None = None):
        self.config = config
        self.index_bundle = index_bundle or load_indexes(config.index_dir)
        self.router = QueryRouter()
        self.text_retriever = HybridTextRetriever(self.index_bundle, config)
        self.image_retriever = ImageRetriever(self.index_bundle, config)
        self.fusion = LateFusion(config)
        self.reranker = LightweightReranker()
        self.generator = ExtractiveGenerator()

    @classmethod
    def from_paths(cls, index_dir: Path, data_dir: Path | None = None) -> "MedicalRAGPipeline":
        config = RAGConfig(index_dir=index_dir, data_dir=data_dir or Path("data")).resolved()
        return cls(config)

    def run(
        self,
        query: str,
        image_path: str | None = None,
        dataset_hint: str | None = None,
        top_k: int | None = None,
    ) -> GeneratedAnswer:
        intent = self.router.route(query, image_path=image_path, dataset_hint=dataset_hint)
        dataset_filter = intent.dataset_hint if intent.dataset_hint else None

        if self.config.force_text_only:
            intent.use_image_branch = False
            intent.modality = Modality.TEXT
            intent.reasons.append(f"profile={self.config.profile_name}: force text-only")
        elif self.config.force_image_for_image_queries and intent.use_image_branch:
            intent.reasons.append(f"profile={self.config.profile_name}: preserve image branch")

        text_results = self.text_retriever.search(
            query,
            top_k=top_k or self.config.text_top_k,
            dataset_filter=dataset_filter,
        )
        image_results = []
        if intent.use_image_branch and not self.config.force_text_only:
            image_results = self.image_retriever.search(
                query,
                image_path=image_path,
                top_k=self.config.image_top_k,
                dataset_filter=dataset_filter,
            )

        evidence = self.fusion.fuse(
            text_results,
            image_results,
            prefer_image=intent.use_image_branch,
        )
        if self.config.enable_rerank:
            evidence = self.reranker.rerank(query, evidence)[: self.config.final_top_k]

        if intent.use_image_branch:
            first_image_idx = next((idx for idx, item in enumerate(evidence) if item.modality == Modality.IMAGE), -1)
            if first_image_idx > 0:
                image_item = evidence[first_image_idx]
                evidence = [image_item, *evidence[:first_image_idx], *evidence[first_image_idx + 1 :]]
            elif first_image_idx < 0 and image_results:
                image_evidence = self.fusion.fuse([], image_results, prefer_image=True)[:1]
                if image_evidence:
                    retained = [item for item in evidence if item.id != image_evidence[0].id]
                    evidence = [image_evidence[0], *retained][: self.config.final_top_k]

        if not intent.use_image_branch:
            # Explicit fallback to text evidence when image branch is not useful/requested.
            evidence = [item for item in evidence if item.modality == Modality.TEXT] or evidence
        else:
            has_image_evidence = any(item.modality == Modality.IMAGE for item in evidence)
            if not has_image_evidence:
                intent.reasons.append("image evidence weak or unavailable; fallback to text evidence")

        return self.generator.generate(intent, evidence)
