"""BioCLIP medical image/text encoder.

Wraps microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 for:
- Image embedding: PIL.Image → 512-dim vector
- Text embedding:  str → 512-dim vector

Reference: MMed-RAG (src/support_repo/MMedRAG_project) uses medical CLIP
for domain-specific image retrieval. BioCLIP provides better domain alignment
than generic CLIP for radiology and pathology images.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default model — can be overridden via config
DEFAULT_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_DIM = 512


class BioCLIPEncoder:
    """BioCLIP image/text encoder with lazy loading and GPU/CPU auto-detection."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load(self) -> None:
        """Lazy-load model and processor on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from open_clip import create_model_from_pretrained, get_tokenizer

            logger.info(f"Loading BioCLIP model: {self.model_name} on {self.device}")
            self._model, self._processor = create_model_from_pretrained(
                f"hf-hub:{self.model_name}"
            )
            self._tokenizer = get_tokenizer(f"hf-hub:{self.model_name}")
            self._model = self._model.to(self.device).eval()
            logger.info(f"BioCLIP loaded successfully on {self.device}")

        except ImportError as exc:
            raise RuntimeError(
                "BioCLIP dependencies missing. Install: "
                "pip install open_clip_torch torch Pillow"
            ) from exc
        except Exception as exc:
            logger.error(f"Failed to load BioCLIP: {exc}")
            raise

    def encode_image(self, image: Any) -> np.ndarray:
        """Encode a single PIL Image to a 512-dim vector.

        Args:
            image: PIL.Image object

        Returns:
            np.ndarray of shape (512,)
        """
        import torch

        self._load()
        image_input = self._processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def encode_image_batch(self, images: list[Any]) -> np.ndarray:
        """Encode a batch of PIL Images.

        Args:
            images: list of PIL.Image objects

        Returns:
            np.ndarray of shape (N, 512)
        """
        import torch

        self._load()
        tensors = torch.stack([self._processor(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string to a 512-dim vector.

        Args:
            text: input string

        Returns:
            np.ndarray of shape (512,)
        """
        import torch

        self._load()
        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def encode_text_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of text strings.

        Args:
            texts: list of strings

        Returns:
            np.ndarray of shape (N, 512)
        """
        import torch

        self._load()
        tokens = self._tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def similarity(self, image: Any, text: str) -> float:
        """Compute cosine similarity between an image and text.

        Args:
            image: PIL.Image
            text: query string

        Returns:
            float similarity score in [-1, 1]
        """
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(text)
        return float(np.dot(img_emb, txt_emb))
