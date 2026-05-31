# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] — In Progress

### Added
- **LangGraph Agent System** (`src/medical_rag/agents/`) — Phase 3
  - Decomposer, text retriever, visual retriever, decision, graph nodes
  - `AgenticRAGPipeline` wrapper class
  - Conditional routing: text / visual / hybrid
- **Model Wrappers** (`src/medical_rag/models/`) — Phase 1
  - BioCLIP encoder (512-dim), BioMedBERT encoder (1024-dim)
  - Qwen2.5-VL VLM (local 4-bit + API fallback)
  - BGE-reranker-v2-m3 cross-encoder
- **Qdrant Indexing** (`src/medical_rag/ingestion/`) — Phase 2
  - Sentence-level text chunker (sliding window)
  - 5-patch image chunker (full + 4 quadrants)
  - Multi-vector Qdrant indexer (in-memory fallback)
- **Extended RAGConfig** with Qdrant, model, and agent settings
- Optional dependency groups in `pyproject.toml` (gpu, qdrant, agent, eval)

## [0.2.0] — 2026-05-28

### Added
- PathVQA dataset support (1GB subset, 3,600 records)
- Canonical JSONL manifest-driven dataset layout for all 6 datasets
- Dataset download + canonicalize CLI commands
- Expanded eval cases and ablation study results
- Status report with implementation priorities

### Changed
- Data loader now tries canonical → processed → raw fallback order

## [0.1.0] — 2026-05-25

### Added
- **Baseline RAG pipeline** (BM25 + TF-IDF + extractive generator)
- Query router (language/modality/dataset detection)
- Hybrid text retriever, caption-based image retriever
- Late fusion (weighted RRF), lexical overlap reranker
- Evaluation pipeline (Recall@k, MRR, routing accuracy, error analysis)
- Ablation profiles A/B/C/D
- Static HTML demo export
- CLI with 12 commands
- 8 smoke tests
- Documentation (project_description.md, dataset_standardization.md)
