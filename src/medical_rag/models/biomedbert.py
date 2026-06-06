"""BioMedBERT biomedical text encoder.

Wraps a HuggingFace BioMedBERT checkpoint for:
- Text embedding: str → vector (mean pooling)

Outperforms generic BERT/RoBERTa on medical text retrieval tasks.
Reference: MMed-RAG uses domain-specific text embeddings for medical QA retrieval.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = os.environ.get(
    "BIOMEDBERT_MODEL",
    "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
)
EMBEDDING_DIM = int(os.environ.get("BIOMEDBERT_DIM", "1024"))


class BioMedBERTEncoder:
    """BioMedBERT encoder for biomedical text with lazy loading and batch support."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self._device = device
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
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading BioMedBERT: {self.model_name} on {self.device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device).eval()
            logger.info("BioMedBERT loaded successfully")
        except ImportError as exc:
            raise RuntimeError(
                "BioMedBERT dependencies missing. Install: pip install transformers torch"
            ) from exc

    def _mean_pool(self, token_embeddings: Any, attention_mask: Any) -> Any:
        """Mean pooling over token embeddings with attention mask."""
        import torch

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string to a 1024-dim vector.

        Args:
            text: input biomedical text
            normalize: L2-normalize the output vector

        Returns:
            np.ndarray embedding vector
        """
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts in batches.

        Args:
            texts: list of input strings
            batch_size: number of texts per forward pass
            normalize: L2-normalize each output vector

        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        import torch

        self._load()
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**encoded)
                embeddings = self._mean_pool(
                    outputs.last_hidden_state, encoded["attention_mask"]
                )
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
