# Medical Multimodal Fine-grained RAG Agent

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-in%20development-orange)](./ROADMAP.md)
[![Tests](https://img.shields.io/badge/tests-8%20passed-green)](./tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

> **Đồ án 2 — Chương trình Kỹ sư AI**  
> Multimodal Fine-grained RAG-based AI Agent cho domain y tế: nhận query text + ảnh X-quang/MRI/pathology, thực hiện retrieval đa phương thức, fuse evidence, sinh câu trả lời có citation.

---

## Trạng thái hiện tại

| Phase | Mô tả | Trạng thái |
|---|---|---|
| **Baseline RAG** | BM25 + TF-IDF + extractive generator | ✅ Done |
| **Data pipeline** | 6 datasets, 69K docs, 49K images, canonical JSONL | ✅ Done |
| **Evaluation** | Recall@k, MRR, routing accuracy, ablation A/B/C/D | ✅ Done |
| **Model wrappers** | BioCLIP, BioMedBERT, Qwen2.5-VL, BGE Reranker | ✅ Done |
| **Qdrant indexing** | Multi-vector indexer, 5-patch image chunking | ✅ Done |
| **LangGraph Agent** | Decomposer → Retriever → Decision → Generator | ✅ Done |
| **Advanced eval** | RAGAS, exact/F1 metrics | 🔄 In Progress |
| **Gradio Demo** | Interactive UI với image upload | ⏳ Planned |
| **Full benchmark** | 200 VQA-RAD + 100 MedQA + 50 MIMIC-CXR | ⏳ Planned |

---

## Kiến trúc tổng thể

```
Query (text + image)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1 — Data & Knowledge Base                    │
│  MedQA | BioASQ | VQA-RAD | ROCO | MIMIC-CXR |     │
│  PathVQA → Canonical JSONL → data/<dataset>/        │
└──────────────────────┬──────────────────────────────┘
                       │ ingest
┌──────────────────────▼──────────────────────────────┐
│  LAYER 2 — Multimodal Indexing                      │
│  BioMedBERT (1024-dim) → Qdrant text_chunks         │
│  BioCLIP (512-dim) + 5-patch → Qdrant image_patches │
│  Fallback: BM25 + TF-IDF → joblib index            │
└──────────────────────┬──────────────────────────────┘
                       │ query
┌──────────────────────▼──────────────────────────────┐
│  LAYER 3 — LangGraph Agent                         │
│  decompose_query (VLM / rule-based router)          │
│    ├── retrieve_text: Dense + BM25 + BGE rerank     │
│    └── retrieve_visual: BioCLIP → ROI crop → score  │
│  fuse_and_verify + generate_answer (Qwen2.5-VL)     │
└──────────────────────┬──────────────────────────────┘
                       │ evaluate
┌──────────────────────▼──────────────────────────────┐
│  LAYER 4 — Evaluation                               │
│  Recall@k | MRR | Routing accuracy | RAGAS          │
│  Ablation A/B/C/D | Error analysis                  │
└─────────────────────────────────────────────────────┘
```

---

## Cài đặt nhanh

### Requirements

- Python ≥ 3.10
- (Optional) CUDA GPU ≥ 16GB cho Qwen2.5-VL local inference
- (Optional) Docker cho Qdrant persistent storage

### Baseline (không cần GPU)

```powershell
# Clone và cài package
git clone <repo-url>
cd KS_Project_2
python -m pip install -e .

# Build index từ data đã có
python -m medical_rag build-index --data-dir data --index-dir data/processed/indexes

# Query thử
python -m medical_rag query "pneumonia treatment" --index-dir data/processed/indexes --top-k 5
```

### Full system (có GPU + Docker)

```powershell
# Cài full dependencies
python -m pip install -e ".[gpu,qdrant,agent,eval]"

# Start Qdrant
docker run -d -p 6333:6333 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant

# Build Qdrant index (cần model downloads lần đầu)
python -m medical_rag build-qdrant-index --data-dir data --limit 500
```

---

## Recommended deployment for RTX 3050 4GB

For the target laptop profile, do **not** self-host Qwen-VL or build full embeddings locally.
Use the split workflow below:

| Runtime | Responsibility |
|---|---|
| **Kaggle / Colab** | BioMedBERT/BioCLIP embedding, BGE reranking, Qdrant index build, benchmark/ablation |
| **Laptop** | Gradio/CLI QA app, local smoke tests, baseline experiments |
| **OpenRouter** | Gemini 2.5 Flash VLM/LLM generation and visual grounding |
| **Qdrant** | Local/in-memory debug or Qdrant Cloud persistent vector store |

Key commands:

```powershell
# local diagnostics
python -m medical_rag test-encoders --mock
python -m medical_rag test-qdrant --qdrant-url :memory:
python -m medical_rag test-openrouter

# Kaggle/Colab real indexing
python scripts/colab_workflow.py build-index-resumable --data-dir data --qdrant-url $QDRANT_URL --datasets all --modality both --image-mode full_only --max-minutes 100 --use-cloud-auth

# app on laptop
python demo/app.py --use-agent --use-qdrant
```

Full runbooks:

- [docs/02-development/deployment.md](docs/02-development/deployment.md)
- [docs/02-development/kaggle-workflow.md](docs/02-development/kaggle-workflow.md)

---

## Sử dụng

### CLI commands

```powershell
# Audit datasets
python -m medical_rag audit-data --data-dir data

# Build index
python -m medical_rag build-index --data-dir data --index-dir data/processed/indexes --limit 500

# Query
python -m medical_rag query "chest xray pneumonia" --index-dir data/processed/indexes --top-k 5

# Evaluate
python -m medical_rag evaluate --eval-file data/eval_cases.json --index-dir data/processed/indexes

# Ablation study
python -m medical_rag ablate --eval-file data/eval_cases.json --output-dir outputs/ablation

# Export static demo
python -m medical_rag export-demo --output-dir demo
```

### Python API — Baseline pipeline

```python
from medical_rag.pipeline import MedicalRAGPipeline
from medical_rag.config import RAGConfig
from medical_rag.indexing import load_indexes

config = RAGConfig(index_dir="data/processed/indexes").resolved()
pipeline = MedicalRAGPipeline(config)
answer = pipeline.run("What treats pneumonia?")
print(answer.answer)
```

### Python API — Agent pipeline

```python
from medical_rag.agents.graph import AgenticRAGPipeline

pipeline = AgenticRAGPipeline({
    "use_qdrant": True,         # requires Qdrant running
    "use_vlm_generation": True, # requires GPU or API key
    "use_fine_grained_visual": True,
    "index_dir": "data/processed/indexes",
})
result = pipeline.run("What does this chest X-ray show?", image_path="xray.jpg")
print(result["answer"])
print(result["citations"])
```

---

## Cấu trúc thư mục

```
KS_Project_2/
├── data/                        # Datasets (không commit lên git)
│   ├── medqa/                   # 12,723 docs — text-only QA
│   ├── bioasq/                  # 8,216 docs — biomedical QA
│   ├── vqa_rad/                 # 2,244 — visual QA radiology
│   ├── roco/                    # 12,415 — medical image captions
│   ├── mimic_cxr/               # 30,633 — chest X-ray reports
│   ├── pathvqa/                 # 3,600 — pathology VQA
│   └── processed/               # Legacy processed JSONL + joblib index
│       └── indexes/             # rag_indexes.joblib (baseline index)
├── src/medical_rag/             # Main Python package
│   ├── agents/                  # LangGraph agent nodes (Phase 3)
│   ├── models/                  # BioCLIP, BioMedBERT, Qwen2.5-VL (Phase 1)
│   ├── ingestion/               # Chunker + Qdrant indexer (Phase 2)
│   ├── retrieval/               # Baseline BM25+TF-IDF retrievers
│   ├── data_tools/              # Dataset canonicalization
│   ├── pipeline.py              # Baseline MedicalRAGPipeline
│   ├── config.py                # RAGConfig (all settings)
│   ├── schema.py                # Pydantic data models
│   └── __main__.py              # CLI entrypoint
├── docs/                        # Documentation
├── tests/                       # Pytest smoke tests
├── scripts/                     # Dataset download utilities
├── outputs/                     # Evaluation results, ablation JSONs
├── demo/                        # Static HTML demo export
└── pyproject.toml               # Package config + dependencies
```

Xem thêm: [docs/02-development/folder-structure.md](./docs/02-development/folder-structure.md)

---

## Datasets

| Dataset | Records | Modality | Source |
|---|---:|---|---|
| MedQA | 12,723 | Text | Local JSONL |
| BioASQ | 8,216 | Text | HuggingFace |
| VQA-RAD | 2,244 | Text + Image | HuggingFace |
| ROCO | 12,415 | Image (caption) | Local 2.5GB subset |
| MIMIC-CXR | 30,633 | Report + X-ray | HuggingFace (credentialed) |
| PathVQA | 3,600 | Pathology image | HuggingFace (1GB subset) |

---

## Tech stack

| Component | Library | Version |
|---|---|---|
| Agent framework | LangGraph | ≥0.2.28 |
| Vector DB | Qdrant | latest |
| Image embedding | BioCLIP | open_clip_torch |
| Text embedding | BioMedBERT-large | transformers |
| VLM | Qwen2.5-VL-7B-Instruct | 4-bit quant |
| Reranker | BGE-reranker-v2-m3 | transformers |
| Evaluation | RAGAS | ≥0.2 |
| Baseline retrieval | rank-bm25 + scikit-learn | — |

---

## Kết quả hiện tại (baseline)

Kết quả từ `outputs/ablation_expanded/ablation_report.md`:

| Config | Mô tả | Recall@5 |
|---|---|---|
| A | Text-only (TF-IDF) | — |
| B | Text-only + Rerank | — |
| C | Image branch (Weighted RRF) | — |
| D | Full pipeline | — |

*Chưa chạy full benchmark trên real eval set. Xem [ROADMAP.md](./ROADMAP.md).*

---

## Đóng góp

Xem [CONTRIBUTING.md](./CONTRIBUTING.md) và [docs/02-development/coding-standards.md](./docs/02-development/coding-standards.md).

## License

MIT — xem [LICENSE](./LICENSE).

## Tài liệu tham khảo

Papers được tham khảo trong `src/support_repo/`:
- **MMed-RAG** (ICLR 2025): Medical multimodal RAG baseline
- **HM-RAG** (ACM MM 2025): Hierarchical multi-agent RAG
- **VimRAG/VRAG**: Progressive visual retrieval, coarse-to-fine
- **A-MAR**: Adaptive retrieval loop, structured reasoning
