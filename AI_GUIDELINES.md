# AI Guidelines & Project Context

> **Đây là file quan trọng nhất cho AI coding agent.**  
> Đọc file này trước khi viết bất kỳ dòng code nào mới.

---

## 1. Project Overview

**Tên**: Medical Multimodal Fine-grained RAG Agent  
**Mục tiêu**: Hệ thống RAG multimodal cho domain y tế, nhận text + image, thực hiện fine-grained retrieval, sinh câu trả lời có citation.  
**Giai đoạn hiện tại**: Đang code — Phase 5 tiếp theo (Advanced Evaluation + RAGAS).

---

## 2. Hai pipeline tồn tại song song

Project có **HAI pipeline** — AI phải phân biệt rõ:

### Pipeline A — Baseline (đã hoàn thành, production-ready)

```
MedicalRAGPipeline (src/medical_rag/pipeline.py)
  → QueryRouter (router.py)
  → HybridTextRetriever (retrieval/text.py)    # BM25 + TF-IDF
  → ImageRetriever (retrieval/image.py)        # caption TF-IDF
  → LateFusion (retrieval/fusion.py)
  → LightweightReranker (retrieval/rerank.py) # lexical overlap
  → ExtractiveGenerator (generation.py)
```

**Index**: `data/processed/indexes/rag_indexes.joblib` (BM25 + TF-IDF sparse matrix)  
**Khi nào dùng**: Tests, ablation, evaluation — vẫn là backbone chính cho so sánh.

### Pipeline B — Agent (mới tạo Phase 1-3, cần GPU/API để dùng đầy đủ)

```
AgenticRAGPipeline (src/medical_rag/agents/graph.py)
  → decompose_query (agents/decomposer.py)     # VLM or rule-based
  → retrieve_text (agents/text_retriever.py)   # BioMedBERT + Qdrant
  → retrieve_visual (agents/visual_retriever.py) # BioCLIP + VLM ROI
  → fuse_and_verify (agents/decision.py)
  → generate_answer (agents/decision.py)       # Qwen2.5-VL or extractive
```

**Index**: Qdrant (`text_chunks` 1024-dim, `image_patches` 512-dim) hoặc joblib fallback  
**Khi nào dùng**: Khi có GPU + Qdrant; hoặc chạy với fallback mode (no GPU).

---

## 3. Cấu trúc package `src/medical_rag/`

```
src/medical_rag/
├── __init__.py            # Package init
├── __main__.py            # CLI (typer) — 12 commands
├── config.py              # RAGConfig — ALL settings here
├── schema.py              # Pydantic models: DocumentChunk, ImageRecord, etc.
│
├── data_tools/            # Dataset canonicalization utilities
│   └── canonicalize.py   # canonical JSONL + manifest.json
│
├── data_loaders.py        # Load canonical/processed JSONL → DocumentChunk/ImageRecord
├── dataset_audit.py       # Audit data directory structure
├── indexing.py            # Build/load joblib BM25+TF-IDF index (baseline)
│
├── retrieval/             # Baseline retrieval components
│   ├── text.py            # HybridTextRetriever (BM25 + TF-IDF)
│   ├── image.py           # ImageRetriever (caption TF-IDF)
│   ├── fusion.py          # LateFusion (weighted RRF)
│   └── rerank.py          # LightweightReranker (lexical overlap)
│
├── pipeline.py            # MedicalRAGPipeline (baseline, orchestrates retrieval/)
├── generation.py          # ExtractiveGenerator (offline, no LLM)
├── router.py              # QueryRouter (rule-based language/modality detection)
│
├── models/                # [Phase 1] Model wrappers — LAZY LOAD
│   ├── bioclip.py         # BioCLIP image+text encoder (512-dim)
│   ├── biomedbert.py      # BioMedBERT text encoder (1024-dim)
│   ├── qwen_vl.py         # Qwen2.5-VL VLM (local 4-bit or API)
│   └── bge_reranker.py    # BGE cross-encoder reranker
│
├── ingestion/             # [Phase 2] Qdrant indexing
│   ├── chunker.py         # Sentence sliding window + 5-patch image
│   └── indexer.py         # Qdrant multi-vector indexer
│
├── agents/                # [Phase 3] LangGraph agent nodes
│   ├── state.py           # AgentState TypedDict
│   ├── decomposer.py      # decompose_query node
│   ├── text_retriever.py  # retrieve_text node
│   ├── visual_retriever.py # retrieve_visual node (2-stage)
│   ├── decision.py        # fuse_and_verify + generate_answer nodes
│   └── graph.py           # StateGraph + AgenticRAGPipeline
│
├── evaluation.py          # Evaluation metrics (Recall@k, MRR, routing)
├── eval_case_builder.py   # Build eval cases from processed JSONL
├── ablation.py            # Ablation profiles A/B/C/D
├── analysis.py            # Ablation report summarizer
├── reporting.py           # Query debug output formatter
├── project_status.py      # Status report builder
└── demo_export.py         # Static HTML demo exporter
```

---

## 4. Config chính — `RAGConfig`

**Tất cả settings đều trong `config.py`**. Không hardcode bất kỳ path hay param nào.

```python
from medical_rag.config import RAGConfig

# Baseline mode (no GPU, no Qdrant)
config = RAGConfig(
    data_dir="data",
    index_dir="data/processed/indexes",
    # use_qdrant=False, use_agent=False  (defaults)
).resolved()

# Agent mode (Qdrant + GPU)
config = RAGConfig(
    use_qdrant=True,
    qdrant_url="http://localhost:6333",
    use_agent=True,
    use_vlm_generation=True,
    use_fine_grained_visual=True,
).resolved()
```

---

## 5. Data schema chuẩn

Mọi dataset sau khi canonical đều có format:

```json
{
  "dataset": "vqa_rad",
  "split": "train",
  "record_id": "unique-id",
  "question": "...",
  "answer": "...",
  "text": "...",
  "image_path": "vqa_rad/images/train/img.jpg",
  "metadata": {}
}
```

IDs luôn có format: `{dataset}-text-{record_id}` và `{dataset}-image-{record_id}`.

---

## 6. Coding rules — AI phải tuân thủ

### Imports
- **LUÔN** dùng `from __future__ import annotations`
- Import model/GPU libraries **lazy** (bên trong hàm/method, không ở top-level)
- Reason: model không phải lúc nào cũng cần, lazy import giúp baseline vẫn chạy không cần GPU

### Error handling
- Model loading **phải** raise `RuntimeError` với hướng dẫn `pip install` rõ ràng
- **Luôn** có fallback cho Qdrant (nếu fail → dùng joblib baseline)
- **Luôn** có fallback cho VLM (nếu fail → dùng extractive generator)

### Backward compatibility
- **Không** thay đổi interface của `MedicalRAGPipeline.run()` — smoke tests dựa vào nó
- **Không** thay đổi schema `DocumentChunk`, `ImageRecord`, `GeneratedAnswer`
- **Không** xóa hoặc rename CLI commands hiện có

### File organization
- Baseline code ở `src/medical_rag/` root hoặc `retrieval/`
- Agent code ở `src/medical_rag/agents/`
- Model wrappers ở `src/medical_rag/models/`
- Ingestion/indexing ở `src/medical_rag/ingestion/`

### Testing
- Mỗi Phase mới **phải** thêm tests vào `tests/test_smoke.py`
- Smoke tests chạy offline (không cần GPU, không cần Qdrant, không cần network)
- Dùng `tmp_path` fixture của pytest cho mọi file I/O

---

## 7. Dataset strategy

| Dataset | Modality | Branch | Index có |
|---|---|---|---|
| medqa | Text | text-only | ✅ joblib |
| bioasq | Text | text-only | ✅ joblib |
| vqa_rad | Text + Image | hybrid | ✅ joblib |
| roco | Image/caption | image | ✅ joblib |
| mimic_cxr | Report + X-ray | hybrid | ✅ joblib |
| pathvqa | Pathology image | image | ✅ canonical JSONL |

**data/raw_pdfs** — EXCLUDED intentionally, không index, không retrieve.

---

## 8. Evaluation strategy

- **Gold IDs**: format `{dataset}-text-{record_id}` hoặc `{dataset}-image-{record_id}`
- **Alias matching**: evaluation.py có alias map để match qua nhiều ID variants
- **Error categories**: `routing_failure`, `retrieval_failure`, `grounding_failure`, `reasoning_failure`, `alias_only_match`
- **Baseline comparison**: MMed-RAG accuracy trên VQA-RAD = baseline chính cần vượt

---

## 9. Các quyết định kiến trúc quan trọng

### ADR-001: Dùng Qdrant thay Chroma
Qdrant hỗ trợ multi-vector collection tốt hơn, binary quantization, và filtering theo payload.

### ADR-002: Lazy model loading
Tất cả model (BioCLIP, BioMedBERT, Qwen2.5-VL) chỉ load khi cần. Baseline chạy không cần GPU.

### ADR-003: Dual pipeline (baseline + agent)
Baseline pipeline giữ nguyên để ablation/comparison. Agent pipeline là contribution mới.

### ADR-004: 5-patch image chunking
Full image + 4 quadrants (tl/tr/bl/br) per image → fine-grained region retrieval.

### ADR-005: Qdrant in-memory fallback
Nếu Qdrant server không reachable, tự động dùng in-memory mode. Không crash.

---

## 10. Phase còn lại

### Phase 5 (tiếp theo): Advanced Evaluation
- File mới: `src/medical_rag/evaluation_advanced.py`
- RAGAS faithfulness + answer relevancy
- Exact match + F1 answer accuracy
- Per-dataset breakdown

### Phase 6: Gradio Demo
- File mới: `demo/app.py`
- Upload image → query → answer + evidence + ROI visualization

### Phase 7: Full Benchmark
- Chạy 200 VQA-RAD + 100 MedQA + 50 MIMIC-CXR
- Export tables cho thesis Chapter 4
