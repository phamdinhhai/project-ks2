"""Model wrappers for medical multimodal RAG.

Provides:
- BioCLIPEncoder: image + text embeddings (512-dim)
- BioMedBERTEncoder: biomedical text embeddings (1024-dim)
- QwenVLModel: vision-language generation + visual grounding
- BGEReranker: cross-encoder reranking
"""
from __future__ import annotations
