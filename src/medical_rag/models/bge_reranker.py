"""BGE cross-encoder reranker.

Wraps BAAI/bge-reranker-v2-m3 for cross-encoder reranking.
Significantly better than lexical overlap for medical evidence reranking.

Reference: HM-RAG uses cross-encoder reranking to improve retrieval precision
before the decision agent stage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@dataclass
class ScoredPassage:
    """A passage with its cross-encoder reranking score."""
    id: str
    text: str
    score: float
    original_rank: int
    metadata: dict


class BGEReranker:
    """Cross-encoder reranker using BGE-reranker-v2-m3.

    Produces much better precision than lexical overlap reranking
    at the cost of higher latency (N forward passes).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self._device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading BGE reranker: {self.model_name} on {self.device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device).eval()
            logger.info("BGE reranker loaded successfully")
        except ImportError as exc:
            raise RuntimeError(
                "Reranker dependencies missing. Install: pip install transformers torch"
            ) from exc

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Compute relevance scores for query-passage pairs.

        Args:
            query: search query
            passages: list of candidate passages

        Returns:
            list of float scores (higher = more relevant)
        """
        import torch

        self._load()
        all_scores: list[float] = []

        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            pairs = [[query, p] for p in batch]
            encoded = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**encoded).logits
                # BGE reranker outputs a single logit per pair
                scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
                if isinstance(scores, float):
                    scores = [scores]
            all_scores.extend(scores)

        return all_scores

    def rerank(
        self,
        query: str,
        passages: list[dict],
        text_key: str = "text",
        id_key: str = "id",
        top_k: int | None = None,
    ) -> list[ScoredPassage]:
        """Rerank a list of passage dicts by cross-encoder score.

        Args:
            query: search query
            passages: list of dicts with at least id_key and text_key
            text_key: key for passage text in each dict
            id_key: key for passage id in each dict
            top_k: optional number of top results to return

        Returns:
            Sorted list of ScoredPassage (best first)
        """
        if not passages:
            return []

        texts = [str(p.get(text_key, "")) for p in passages]
        scores = self.score_pairs(query, texts)

        scored = [
            ScoredPassage(
                id=str(p.get(id_key, f"passage-{i}")),
                text=texts[i],
                score=scores[i],
                original_rank=i + 1,
                metadata={k: v for k, v in p.items() if k not in (id_key, text_key)},
            )
            for i, p in enumerate(passages)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            scored = scored[:top_k]
        return scored
