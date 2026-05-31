from __future__ import annotations

import numpy as np

from medical_rag.config import RAGConfig
from medical_rag.schema import DocumentChunk, Modality, RetrievalResult
from medical_rag.text_utils import minmax, tokenize


class HybridTextRetriever:
    """Hybrid BM25 + TF-IDF retriever; dense slot can be replaced later."""

    def __init__(self, index_bundle: dict, config: RAGConfig):
        self.documents: list[DocumentChunk] = index_bundle.get("documents", [])
        self.bm25 = index_bundle.get("text_bm25")
        self.vectorizer = index_bundle.get("text_vectorizer")
        self.matrix = index_bundle.get("text_matrix")
        self.config = config

    def search(
        self,
        query: str,
        top_k: int | None = None,
        dataset_filter: str | set[str] | None = None,
    ) -> list[RetrievalResult]:
        if not self.documents:
            return []
        top_k = top_k or self.config.text_top_k
        allowed = {dataset_filter} if isinstance(dataset_filter, str) else set(dataset_filter or [])

        bm25_scores = self.bm25.get_scores(tokenize(query)).tolist() if self.bm25 else [0.0] * len(self.documents)
        dense_scores = [0.0] * len(self.documents)
        if self.vectorizer is not None and self.matrix is not None:
            q_vec = self.vectorizer.transform([query])
            dense_scores = np.asarray((self.matrix @ q_vec.T).todense()).ravel().tolist()

        bm25_norm = minmax([float(score) for score in bm25_scores])
        dense_norm = minmax([float(score) for score in dense_scores])
        combined = [
            self.config.bm25_weight * b + self.config.dense_weight * d
            for b, d in zip(bm25_norm, dense_norm)
        ]
        order = np.argsort(combined)[::-1]

        results: list[RetrievalResult] = []
        for idx in order.tolist():
            doc = self.documents[idx]
            if allowed and doc.dataset not in allowed:
                continue
            rank = len(results) + 1
            results.append(RetrievalResult(
                id=doc.id,
                modality=Modality.TEXT,
                score=float(combined[idx]),
                rank=rank,
                text=doc.text,
                dataset=doc.dataset,
                source_path=doc.source_path,
                metadata={**doc.metadata, "bm25_score": bm25_scores[idx], "dense_score": dense_scores[idx]},
            ))
            if len(results) >= top_k:
                break
        return results
