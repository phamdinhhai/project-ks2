# Folder Structure — Full Annotated Map

Every file in `src/medical_rag/` with purpose and status.

## Root files

| File | Purpose | Status |
|---|---|---|
| `README.md` | Quick start, architecture overview | ✅ Current |
| `AI_GUIDELINES.md` | Context file for AI/dev — **read first** | ✅ Current |
| `ROADMAP.md` | Done/todo/planned features | ✅ Current |
| `CHANGELOG.md` | Version history | ✅ Current |
| `CONTRIBUTING.md` | Coding standards, commit rules | ✅ Current |
| `LICENSE` | MIT License | ✅ Current |
| `pyproject.toml` | Package config, dependencies | ✅ Current |
| `.gitignore` | Git exclusions | ✅ Current |

## `src/medical_rag/` — Core package

### Root modules (baseline pipeline)

| File | Purpose | Lines | Used by |
|---|---|---:|---|
| `__init__.py` | Package metadata | 8 | import |
| `__main__.py` | CLI entrypoint (typer, 12 commands) | 195 | `python -m medical_rag` |
| `config.py` | `RAGConfig` — all runtime settings | 51 | Everything |
| `schema.py` | Pydantic models: Language, Modality, DocumentChunk, etc. | 75 | Everything |
| `pipeline.py` | `MedicalRAGPipeline` — baseline orchestrator | 93 | CLI, tests, evaluation |
| `router.py` | `QueryRouter` — rule-based route (lang/modality/dataset) | 78 | pipeline.py, agents/decomposer.py |
| `generation.py` | `ExtractiveGenerator` — offline answer from evidence | 43 | pipeline.py |
| `text_utils.py` | normalize_text, tokenize, minmax, lexical_overlap | 33 | retrieval, reranker |
| `data_loaders.py` | Load canonical/processed JSONL → DocumentChunk/ImageRecord | 298 | indexing.py |
| `indexing.py` | Build/load joblib index (BM25 + TF-IDF) | 119 | CLI, pipeline |
| `evaluation.py` | Evaluate pipeline: Recall@k, MRR, routing, errors | 206 | CLI, ablation |
| `ablation.py` | Run profiles A/B/C/D ablation study | 85 | CLI |
| `analysis.py` | Summarize ablation → markdown report | 90 | CLI |
| `eval_case_builder.py` | Build eval cases from processed JSONL | 187 | CLI |
| `dataset_audit.py` | Audit data directory structure | 126 | CLI, status |
| `project_status.py` | Status report builder | 52 | CLI |
| `reporting.py` | Query debug output formatter | 50 | CLI |
| `demo_export.py` | Static HTML demo exporter | 110 | CLI |

### `retrieval/` — Baseline retrieval components

| File | Purpose | Class |
|---|---|---|
| `text.py` | BM25 + TF-IDF hybrid text search | `HybridTextRetriever` |
| `image.py` | Caption TF-IDF image search | `ImageRetriever` |
| `fusion.py` | Weighted RRF late fusion | `LateFusion` |
| `rerank.py` | Lexical overlap reranker | `LightweightReranker` |

### `models/` — [Phase 1] Model wrappers (lazy-loaded)

| File | Model | Dim | GPU needed |
|---|---|---:|---|
| `bioclip.py` | BioCLIP image+text | 512 | ✅ (or CPU slow) |
| `biomedbert.py` | BioMedBERT text | 1024 | ✅ (or CPU slow) |
| `qwen_vl.py` | Qwen2.5-VL-7B VLM | — | ✅ or API |
| `bge_reranker.py` | BGE cross-encoder | — | ✅ (or CPU slow) |

### `ingestion/` — [Phase 2] Qdrant indexing

| File | Purpose |
|---|---|
| `chunker.py` | Text: sentence sliding window; Image: 5-patch (full + 4 quadrants) |
| `indexer.py` | Qdrant multi-vector index builder (in-memory fallback) |

### `agents/` — [Phase 3] LangGraph agent nodes

| File | Node / Class | Purpose |
|---|---|---|
| `state.py` | `AgentState` | TypedDict for LangGraph state |
| `decomposer.py` | `decompose_query()` | Query → text_subquery + visual_subquery |
| `text_retriever.py` | `retrieve_text()` | Dense/BM25 hybrid + BGE rerank |
| `visual_retriever.py` | `retrieve_visual()` | Stage 1 coarse + Stage 2 ROI |
| `decision.py` | `fuse_and_verify()` + `generate_answer()` | Fusion + VLM generation |
| `graph.py` | `AgenticRAGPipeline` | StateGraph assembly + wrapper |

### `data_tools/` — Dataset utilities

| File | Purpose |
|---|---|
| `canonicalize.py` | Convert HF/raw datasets → canonical JSONL + manifest.json |

## `data/` — Datasets (excluded from git)

```
data/
├── medqa/           # 12,723 rows — local JSONL
├── bioasq/          # 8,216 rows — HF download
├── vqa_rad/         # 2,244 rows — HF download + images/
├── roco/            # 12,415 rows — local 2.5GB subset
├── mimic_cxr/       # 30,633 rows — HF download + images/
├── pathvqa/         # 3,600 rows — HF download + images/
├── processed/       # Legacy processed JSONL files
│   └── indexes/     # rag_indexes.joblib (baseline index)
└── raw_pdfs/        # EXCLUDED from indexing (out of scope)
```

## `src/support_repo/` — Reference papers & repos (READ-ONLY)

```
support_repo/
├── MMedRAG_project/  # MMed-RAG paper + repo — baseline comparison
├── HMRAG_project/    # HM-RAG paper + repo — agent architecture reference
├── VimRAG_project/   # VimRAG paper + repo — fine-grained visual retrieval
└── A-MAR_project/    # A-MAR paper + repo — adaptive retrieval
```

> ⚠️ These are **reference only**. Never import from `support_repo/` in production code.

## Other folders

| Folder | Purpose | Git tracked |
|---|---|---|
| `outputs/` | Eval results, ablation JSONs, debug output | Partially |
| `demo/` | Static HTML demo export | ✅ |
| `tests/` | Pytest smoke tests | ✅ |
| `scripts/` | Dataset download utility | ✅ |
