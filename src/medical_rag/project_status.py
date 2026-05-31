from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_rag.dataset_audit import audit_data_directory
from medical_rag.indexing import index_stats, load_indexes

IMPLEMENTATION_PRIORITIES = [
    {"id": 1, "requirement": "Build query router", "status": "done", "module": "medical_rag.router"},
    {"id": 2, "requirement": "Plug in text hybrid retrieval", "status": "done", "module": "medical_rag.retrieval.text"},
    {"id": 3, "requirement": "Add image retrieval branch", "status": "done", "module": "medical_rag.retrieval.image"},
    {"id": 4, "requirement": "Add late fusion", "status": "done", "module": "medical_rag.retrieval.fusion"},
    {"id": 5, "requirement": "Add reranking", "status": "done", "module": "medical_rag.retrieval.rerank"},
    {"id": 6, "requirement": "Add evaluation scripts", "status": "done", "module": "medical_rag.evaluation"},
]


def build_status_report(data_dir: Path, index_dir: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "project": "medical-multimodal-rag",
        "scope": {
            "active_datasets": ["medqa", "bioasq", "vqa_rad", "roco", "mimic_cxr"],
            "excluded_sources": ["raw_pdfs"],
            "raw_pdfs_used_for_indexing": False,
        },
        "implementation_priorities": IMPLEMENTATION_PRIORITIES,
        "current_stage": "offline-first multimodal RAG baseline with evaluation and ablation support",
        "known_limitations": [
            "Text dense branch is currently TF-IDF baseline, not neural biomedical embeddings.",
            "Image branch uses captions/metadata text, not true visual encoder embeddings yet.",
            "Generation is extractive/evidence summarization, not LLM reasoning yet.",
        ],
        "recommended_next_steps": [
            "Run ablation configs A/B/C/D on expanded eval cases.",
            "Add neural biomedical text embeddings.",
            "Add visual encoder retrieval for medical images.",
            "Integrate citation-grounded LLM generation.",
        ],
    }

    if data_dir.exists():
        report["data_audit"] = audit_data_directory(data_dir, sample_size=1)

    if index_dir and (index_dir / "rag_indexes.joblib").exists():
        bundle = load_indexes(index_dir)
        report["index_stats"] = index_stats(bundle)
    else:
        report["index_stats"] = {"status": "not_found", "index_dir": str(index_dir) if index_dir else None}

    return report
