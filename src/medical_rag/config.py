from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """Runtime configuration for the offline-first medical RAG pipeline."""

    data_dir: Path = Field(default=Path("data"))
    index_dir: Path = Field(default=Path("data/processed/indexes"))
    text_top_k: int = 8
    image_top_k: int = 5
    final_top_k: int = 6
    bm25_weight: float = 0.55
    dense_weight: float = 0.45
    image_weight: float = 0.35
    rrf_k: int = 60
    min_image_score: float = 0.02
    enable_rerank: bool = True
    profile_name: str = "D"
    force_text_only: bool = False
    force_image_for_image_queries: bool = True

    # --- Qdrant settings ---
    use_qdrant: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_text: str = "text_chunks"
    qdrant_collection_image: str = "image_patches"

    # --- Model settings ---
    text_model_name: str = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract-fulltext"
    image_model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    # --- Agent settings ---
    use_agent: bool = False
    use_cross_encoder_rerank: bool = False
    use_vlm_generation: bool = False
    use_fine_grained_visual: bool = False

    # --- Hardware profile ---
    use_mock_models: bool = False        # Use mock models for offline testing
    use_cpu_offload: bool = True         # Offload model layers to RAM when VRAM low
    encoder_batch_size: int = 1          # Batch size for BioCLIP/BioMedBERT (1 = safe for 4GB)
    max_new_tokens: int = 256            # Limit VLM generation (shorter = less VRAM)
    vram_gb: float = 0.0                 # 0 = auto-detect; set manually if needed

    # --- LLM/VLM provider settings ---
    llm_provider: str = "auto"          # auto | openrouter | qwen_api | local | mock
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemini-2.5-flash"

    def resolved(self, root: Path | None = None) -> "RAGConfig":
        base = root or Path.cwd()
        copy = self.model_copy(deep=True)
        if not copy.data_dir.is_absolute():
            copy.data_dir = base / copy.data_dir
        if not copy.index_dir.is_absolute():
            copy.index_dir = base / copy.index_dir
        return copy
