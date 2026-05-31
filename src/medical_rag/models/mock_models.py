"""Mock model implementations for offline testing.

Allows full agent pipeline testing without GPU, internet, or model downloads.
All mocks return deterministic or random-but-valid outputs.

Usage in tests:
    from medical_rag.models.mock_models import MockBioCLIP, MockBioMedBERT, MockQwenVL

Usage in agent:
    pipeline = AgenticRAGPipeline({
        "use_mock_models": True,
        ...
    })
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class MockBioCLIP:
    """Deterministic 512-dim vectors for offline testing.

    Vectors are seeded from input hash so same input = same vector.
    This allows consistent test assertions.
    """

    EMBEDDING_DIM = 512

    def encode_image(self, image: Any) -> np.ndarray:
        """Return deterministic 512-dim vector from image pixel hash."""
        try:
            import io
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            seed = int(hashlib.md5(buf.getvalue()[:64]).hexdigest(), 16) % (2**32)
        except Exception:
            seed = 42
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def encode_image_batch(self, images: list) -> np.ndarray:
        return np.stack([self.encode_image(img) for img in images])

    def encode_text(self, text: str) -> np.ndarray:
        """Return deterministic 512-dim vector from text hash."""
        seed = int(hashlib.md5(text.encode()[:64]).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def encode_text_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode_text(t) for t in texts])

    def similarity(self, image: Any, text: str) -> float:
        img_vec = self.encode_image(image)
        txt_vec = self.encode_text(text)
        return float(np.dot(img_vec, txt_vec))


class MockBioMedBERT:
    """Deterministic 1024-dim vectors for offline testing."""

    EMBEDDING_DIM = 1024

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()[:64]).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
        if normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        return np.stack([self.encode(t, normalize=normalize) for t in texts])


class MockQwenVL:
    """Template-based answer generator for offline testing.

    Returns structured answers that exercise the full pipeline
    without needing GPU or network.
    """

    def generate(
        self,
        question: str,
        image: Any | None = None,
        context: str | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        ctx_preview = context[:300].replace("\n", " ") if context else "no context provided"
        image_note = " [Image analysis included.]" if image is not None else ""
        return (
            f"[MOCK ANSWER]{image_note}\n"
            f"Question: {question}\n"
            f"Based on retrieved medical evidence: {ctx_preview}...\n"
            f"Conclusion: This is a mock response for pipeline testing. "
            f"Deploy with real Qwen2.5-VL for actual inference."
        )

    def ground_region(
        self,
        question: str,
        image: Any,
    ) -> dict:
        """Return a plausible center region as mock grounding."""
        q_hash = int(hashlib.md5(question.encode()[:32]).hexdigest(), 16) % 100
        offset = q_hash / 400.0  # small random offset
        return {
            "bbox": [0.2 + offset, 0.2 + offset, 0.8 - offset, 0.8 - offset],
            "confidence": 0.65 + offset,
            "description": f"[MOCK] Relevant region for: {question[:60]}",
        }


class MockBGEReranker:
    """Lexical-overlap based mock reranker for offline testing."""

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        query_words = set(query.lower().split())
        scores = []
        for p in passages:
            p_words = set(p.lower().split())
            overlap = len(query_words & p_words) / max(1, len(query_words))
            scores.append(float(overlap))
        return scores

    def rerank(
        self,
        query: str,
        passages: list[dict],
        text_key: str = "text",
        id_key: str = "id",
        top_k: int | None = None,
    ) -> list:
        from medical_rag.models.bge_reranker import ScoredPassage

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
        return scored[:top_k] if top_k else scored
