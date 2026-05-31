"""Visual retrieval agent node — 2-stage fine-grained retrieval.

Stage 1 (Coarse): BioCLIP text-guided search → top-10 images via Qdrant.
Stage 2 (Fine-grained): Qwen2.5-VL ground_region → crop ROI → re-encode → top-3.

Reference:
- VimRAG (src/support_repo/VimRAG_project): progressive coarse→fine visual retrieval
- A-MAR: adaptive retrieval loop with quality check
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from medical_rag.agents.state import AgentState

logger = logging.getLogger(__name__)


def retrieve_visual(state: AgentState) -> AgentState:
    """LangGraph node: 2-stage fine-grained visual retrieval.

    Stage 1: BioCLIP coarse search → top-10 full images.
    Stage 2: If uploaded image + VLM available, run ground_region + ROI crop.

    Falls back to baseline caption-based image retrieval if Qdrant unavailable.
    """
    query = state.get("visual_subquery") or state.get("question", "")
    image_path_str = state.get("image_path")
    step_count = state.get("step_count", 0) + 1
    cfg = state.get("config", {})
    steps = list(state.get("reasoning_steps", []))

    use_qdrant = cfg.get("use_qdrant", False)
    use_fine_grained = cfg.get("use_fine_grained_visual", False)
    top_k = cfg.get("image_top_k", 5)
    dataset_hint = state.get("dataset_hint")

    coarse_results: list[dict[str, Any]] = []

    # Stage 1: Coarse retrieval
    if use_qdrant:
        try:
            coarse_results = _coarse_search_qdrant(
                query, image_path_str, cfg, top_k * 2, dataset_hint
            )
            steps.append(
                f"[visual_retrieval] Stage 1 Qdrant dense → {len(coarse_results)} coarse results"
            )
        except Exception as exc:
            logger.warning(f"Qdrant visual search failed: {exc}")
            steps.append(f"[visual_retrieval] Qdrant failed ({exc}), falling back")

    if not coarse_results:
        coarse_results = _coarse_search_baseline(query, cfg, top_k, dataset_hint)
        steps.append(
            f"[visual_retrieval] Stage 1 caption baseline → {len(coarse_results)} coarse results"
        )

    # Stage 2: Fine-grained ROI if conditions met
    if use_fine_grained and image_path_str and coarse_results:
        try:
            fine_results = _fine_grained_roi(
                query, image_path_str, coarse_results, cfg, top_k
            )
            if fine_results:
                steps.append(
                    f"[visual_retrieval] Stage 2 ROI fine-grained → {len(fine_results)} results"
                )
                return {
                    "visual_evidence": fine_results,
                    "step_count": step_count,
                    "reasoning_steps": steps,
                }
        except Exception as exc:
            logger.warning(f"Fine-grained visual retrieval failed: {exc}")
            steps.append(f"[visual_retrieval] Stage 2 failed ({exc}), using coarse results")

    final = coarse_results[:top_k]
    return {
        "visual_evidence": final,
        "step_count": step_count,
        "reasoning_steps": steps,
    }


def _coarse_search_qdrant(
    query: str,
    image_path: str | None,
    cfg: dict[str, Any],
    top_k: int,
    dataset_hint: str | None,
) -> list[dict[str, Any]]:
    """BioCLIP coarse image search via Qdrant."""
    from medical_rag.ingestion.indexer import IMAGE_COLLECTION, get_client

    if cfg.get("use_mock_models"):
        from medical_rag.models.mock_models import MockBioCLIP
        encoder = MockBioCLIP()
    else:
        from medical_rag.models.bioclip import BioCLIPEncoder
        encoder = BioCLIPEncoder(model_name=cfg.get("image_model_name", ""))

    if cfg.get("use_cloud_auth") or cfg.get("qdrant_api_key"):
        from medical_rag.ingestion.qdrant_cloud import get_qdrant_cloud_client
        client = get_qdrant_cloud_client(
            url=cfg.get("qdrant_url"),
            api_key=cfg.get("qdrant_api_key"),
        )
    else:
        client = get_client(cfg.get("qdrant_url"))

    # If uploaded image available, blend image + text embeddings
    if image_path and Path(image_path).exists():
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img_vec = encoder.encode_image(img)
        txt_vec = encoder.encode_text(query)
        query_vector = (0.6 * img_vec + 0.4 * txt_vec).tolist()
    else:
        query_vector = encoder.encode_text(query).tolist()

    query_filter = None
    if dataset_hint:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        query_filter = Filter(
            must=[FieldCondition(key="dataset", match=MatchValue(value=dataset_hint))]
        )

    results = client.search(
        collection_name=cfg.get("qdrant_collection_image", IMAGE_COLLECTION),
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )
    return [
        {
            "id": r.payload.get("image_id", ""),
            "text": r.payload.get("caption", ""),
            "score": float(r.score),
            "dataset": r.payload.get("dataset", ""),
            "image_path": r.payload.get("image_path"),
            "patch_id": r.payload.get("patch_id", "full"),
            "bbox": r.payload.get("bbox", [0.0, 0.0, 1.0, 1.0]),
            "record_id": r.payload.get("record_id", ""),
            "modality": "image",
            "source": "qdrant_bioclip",
        }
        for r in results
    ]


def _coarse_search_baseline(
    query: str,
    cfg: dict[str, Any],
    top_k: int,
    dataset_hint: str | None,
) -> list[dict[str, Any]]:
    """Fallback caption-based image search from joblib index."""
    from pathlib import Path as _Path

    from medical_rag.config import RAGConfig
    from medical_rag.indexing import load_indexes
    from medical_rag.retrieval.image import ImageRetriever

    index_dir = _Path(cfg.get("index_dir", "data/processed/indexes"))
    if not (index_dir / "rag_indexes.joblib").exists():
        return []

    rag_config = RAGConfig(
        index_dir=index_dir,
        image_top_k=top_k,
        min_image_score=cfg.get("min_image_score", 0.01),
    )
    bundle = load_indexes(index_dir)
    retriever = ImageRetriever(bundle, rag_config)
    results = retriever.search(query, top_k=top_k, dataset_filter=dataset_hint)
    return [
        {
            "id": r.id,
            "text": r.text,
            "score": r.score,
            "dataset": r.dataset,
            "image_path": r.source_path,
            "patch_id": "full",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "record_id": r.metadata.get("record_id", ""),
            "modality": "image",
            "source": "baseline_caption_tfidf",
        }
        for r in results
    ]


def _fine_grained_roi(
    query: str,
    image_path_str: str,
    coarse_results: list[dict[str, Any]],
    cfg: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Stage 2: Qwen2.5-VL ground_region on coarse results → crop → re-score.

    For each candidate image, VLM identifies the most relevant sub-region,
    which is then cropped and re-encoded with BioCLIP for final scoring.
    """
    from PIL import Image

    import os

    if cfg.get("use_mock_models") or cfg.get("llm_provider") == "mock":
        from medical_rag.models.mock_models import MockBioCLIP, MockQwenVL
        vlm = MockQwenVL()
        encoder = MockBioCLIP()
    elif cfg.get("llm_provider") == "openrouter" or os.environ.get("OPENROUTER_API_KEY"):
        from medical_rag.models.bioclip import BioCLIPEncoder
        from medical_rag.models.openrouter_vlm import OpenRouterVLM
        vlm = OpenRouterVLM(
            model=cfg.get("openrouter_model"),
            base_url=cfg.get("openrouter_base_url"),
            max_tokens=cfg.get("max_new_tokens", 256),
        )
        encoder = BioCLIPEncoder(model_name=cfg.get("image_model_name", ""))
    else:
        from medical_rag.models.bioclip import BioCLIPEncoder
        from medical_rag.models.qwen_vl import QwenVLModel
        vlm = QwenVLModel(
            model_name=cfg.get("vlm_model_name"),
            use_cpu_offload=cfg.get("use_cpu_offload", True),
            max_new_tokens=cfg.get("max_new_tokens", 256),
        )
        encoder = BioCLIPEncoder(model_name=cfg.get("image_model_name", ""))

    query_vec = encoder.encode_text(query)

    refined: list[dict[str, Any]] = []
    for candidate in coarse_results[:10]:  # process top-10 coarse results
        cand_image_path = candidate.get("image_path")
        if not cand_image_path or not Path(cand_image_path).exists():
            continue

        img = Image.open(cand_image_path).convert("RGB")
        width, height = img.size

        # Ground region
        grounding = vlm.ground_region(question=query, image=img)
        bbox = grounding["bbox"]  # [x1, y1, x2, y2] relative
        confidence = grounding["confidence"]

        # Crop the ROI
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        cropped = img.crop((x1, y1, x2, y2))

        # Re-encode cropped region with BioCLIP
        crop_vec = encoder.encode_image(cropped)
        similarity = float((query_vec * crop_vec).sum())

        # Combined score: original + grounding confidence + region similarity
        combined_score = 0.5 * candidate["score"] + 0.3 * confidence + 0.2 * similarity

        refined.append({
            **candidate,
            "bbox": bbox,
            "score": combined_score,
            "grounding_confidence": confidence,
            "grounding_description": grounding.get("description", ""),
            "source": "fine_grained_roi",
        })

    refined.sort(key=lambda x: x["score"], reverse=True)
    return refined[:top_k]
