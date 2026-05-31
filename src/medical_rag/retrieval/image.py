from __future__ import annotations

import numpy as np

from medical_rag.config import RAGConfig
from medical_rag.schema import ImageRecord, Modality, RetrievalResult


class ImageRetriever:
    """Caption/metadata image retriever with a CLIP-compatible interface placeholder."""

    def __init__(self, index_bundle: dict, config: RAGConfig):
        self.images: list[ImageRecord] = index_bundle.get("images", [])
        self.vectorizer = index_bundle.get("image_vectorizer")
        self.matrix = index_bundle.get("image_matrix")
        self.config = config

    def search(
        self,
        query: str,
        image_path: str | None = None,
        top_k: int | None = None,
        dataset_filter: str | set[str] | None = None,
    ) -> list[RetrievalResult]:
        if not self.images or self.vectorizer is None or self.matrix is None:
            return []
        top_k = top_k or self.config.image_top_k
        allowed = {dataset_filter} if isinstance(dataset_filter, str) else set(dataset_filter or [])

        # Offline baseline: retrieve visual records through captions/metadata text.
        # Future extension: blend query embedding with uploaded image embedding here.
        q_vec = self.vectorizer.transform([query or image_path or "medical image"])
        scores = np.asarray((self.matrix @ q_vec.T).todense()).ravel()
        order = np.argsort(scores)[::-1]

        results: list[RetrievalResult] = []
        for idx in order.tolist():
            image = self.images[idx]
            if allowed and image.dataset not in allowed:
                continue
            rank = len(results) + 1
            score = float(scores[idx])
            if score < self.config.min_image_score and rank > 1:
                continue
            results.append(RetrievalResult(
                id=image.id,
                modality=Modality.IMAGE,
                score=score,
                rank=rank,
                text=image.caption,
                dataset=image.dataset,
                source_path=image.image_path,
                metadata={**image.metadata, "image_path": image.image_path, "uploaded_image_path": image_path},
            ))
            if len(results) >= top_k:
                break
        return results
