"""Text retrieval agent node.

Dense (BioMedBERT + Qdrant) + BM25 hybrid retrieval with optional
BGE cross-encoder reranking.

Reference: HM-RAG multi-source retrieval, MMed-RAG medical text retrieval.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from medical_rag.agents.state import AgentState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _cached_load_indexes(index_dir: str) -> dict[str, Any]:
    """Load joblib indexes once per process for agent fallback retrieval."""
    from medical_rag.indexing import load_indexes

    return load_indexes(Path(index_dir))


def retrieve_text(state: AgentState) -> AgentState:
    """LangGraph node: hybrid text retrieval.

    Pipeline:
    1. BioMedBERT encode text_subquery → dense vector
    2. Qdrant search text_chunks → top-20 by cosine
    3. BM25 scoring on results → hybrid score
    4. Optional BGE cross-encoder rerank → top-5
    5. Update state.text_evidence

    Falls back to baseline TF-IDF+BM25 pipeline if Qdrant unavailable.
    """
    query = state.get("text_subquery") or state.get("question", "")
    step_count = state.get("step_count", 0) + 1
    cfg = state.get("config", {})
    steps = list(state.get("reasoning_steps", []))

    use_qdrant = cfg.get("use_qdrant", False)
    top_k = cfg.get("text_top_k", 8)
    dataset_hint = state.get("dataset_hint")

    evidence: list[dict[str, Any]] = []

    if use_qdrant:
        try:
            evidence = _search_qdrant(query, cfg, top_k, dataset_hint)
            steps.append(f"[text_retrieval] Qdrant dense search → {len(evidence)} results")
        except Exception as exc:
            logger.warning(f"Qdrant text search failed: {exc}")
            steps.append(f"[text_retrieval] Qdrant failed ({exc}), falling back to baseline")

    if not evidence:
        evidence = _search_baseline(query, cfg, top_k, dataset_hint)
        steps.append(f"[text_retrieval] Baseline BM25+TF-IDF → {len(evidence)} results")

    # Optional cross-encoder reranking
    if cfg.get("use_cross_encoder_rerank") and evidence:
        try:
            evidence = _rerank_cross_encoder(query, evidence, cfg)
            steps.append(f"[text_retrieval] BGE cross-encoder rerank → {len(evidence)} results")
        except Exception as exc:
            logger.warning(f"Cross-encoder rerank failed: {exc}")
            steps.append(f"[text_retrieval] Rerank failed ({exc}), keeping original order")

    return {
        "text_evidence": evidence,
        "step_count": step_count,
        "reasoning_steps": steps,
    }


def _search_qdrant(
    query: str,
    cfg: dict[str, Any],
    top_k: int,
    dataset_hint: str | None,
) -> list[dict[str, Any]]:
    """Dense retrieval via BioMedBERT + Qdrant."""
    from medical_rag.ingestion.indexer import TEXT_COLLECTION, get_client

    if cfg.get("use_mock_models"):
        from medical_rag.models.mock_models import MockBioMedBERT
        encoder = MockBioMedBERT()
    else:
        from medical_rag.models.biomedbert import BioMedBERTEncoder
        model_name = cfg.get("text_model_name") or None
        encoder = BioMedBERTEncoder(model_name=model_name)

    if cfg.get("use_cloud_auth") or cfg.get("qdrant_api_key"):
        from medical_rag.ingestion.qdrant_cloud import get_qdrant_cloud_client
        client = get_qdrant_cloud_client(
            url=cfg.get("qdrant_url"),
            api_key=cfg.get("qdrant_api_key"),
        )
    else:
        client = get_client(cfg.get("qdrant_url"))
    query_vector = encoder.encode(query, normalize=True).tolist()

    # Build filter if dataset_hint provided
    query_filter = None
    if dataset_hint:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        query_filter = Filter(
            must=[FieldCondition(key="dataset", match=MatchValue(value=dataset_hint))]
        )

    results = client.search(
        collection_name=cfg.get("qdrant_collection_text", TEXT_COLLECTION),
        query_vector=query_vector,
        limit=top_k * 2,  # over-retrieve for reranking
        query_filter=query_filter,
    )
    return [
        {
            "id": r.payload.get("doc_id", ""),
            "text": r.payload.get("text", ""),
            "score": float(r.score),
            "dataset": r.payload.get("dataset", ""),
            "source": "qdrant_dense",
            "record_id": r.payload.get("record_id", ""),
            "modality": "text",
        }
        for r in results
    ]


def _search_baseline(
    query: str,
    cfg: dict[str, Any],
    top_k: int,
    dataset_hint: str | None,
) -> list[dict[str, Any]]:
    """Fallback to baseline BM25+TF-IDF pipeline (joblib index)."""
    from pathlib import Path

    from medical_rag.config import RAGConfig
    from medical_rag.retrieval.text import HybridTextRetriever

    index_dir = Path(cfg.get("index_dir", "data/processed/indexes"))
    if not (index_dir / "rag_indexes.joblib").exists():
        return []

    rag_config = RAGConfig(
        index_dir=index_dir,
        bm25_weight=cfg.get("bm25_weight", 0.55),
        dense_weight=cfg.get("dense_weight", 0.45),
        text_top_k=top_k,
    )
    bundle = _cached_load_indexes(str(index_dir))
    retriever = HybridTextRetriever(bundle, rag_config)
    results = retriever.search(query, top_k=top_k, dataset_filter=dataset_hint)
    return [
        {
            "id": r.id,
            "text": r.text,
            "score": r.score,
            "dataset": r.dataset,
            "source": "baseline_bm25_tfidf",
            "record_id": r.metadata.get("record_id", ""),
            "modality": "text",
        }
        for r in results
    ]


def _rerank_cross_encoder(
    query: str,
    evidence: list[dict[str, Any]],
    cfg: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rerank using BGE cross-encoder."""
    from medical_rag.models.bge_reranker import BGEReranker

    reranker = BGEReranker(model_name=cfg.get("reranker_model_name", "BAAI/bge-reranker-v2-m3"))
    scored = reranker.rerank(query, evidence, text_key="text", id_key="id", top_k=top_k)
    return [
        {
            "id": sp.id,
            "text": sp.text,
            "score": sp.score,
            "dataset": sp.metadata.get("dataset", ""),
            "source": sp.metadata.get("source", "reranked"),
            "record_id": sp.metadata.get("record_id", ""),
            "modality": "text",
        }
        for sp in scored
    ]
