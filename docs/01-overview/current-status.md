# Current Status — What's done, what's not

> Last updated: 2026-05-30

## Fully working (can run now)

### Baseline RAG pipeline
```powershell
python -m medical_rag build-index --data-dir data --index-dir data/processed/indexes --limit 500
python -m medical_rag query "pneumonia treatment" --index-dir data/processed/indexes
python -m medical_rag evaluate --eval-file data/eval_cases.json --index-dir data/processed/indexes
python -m medical_rag ablate --eval-file data/eval_cases.json --output-dir outputs/ablation
```

All 8 smoke tests pass: `pytest tests/test_smoke.py`

### Datasets available locally
- ✅ MedQA (12,723 docs)
- ✅ BioASQ (8,216 docs)
- ✅ VQA-RAD (2,244 docs + images)
- ✅ ROCO (12,415 image records)
- ✅ MIMIC-CXR (30,633 records)
- ✅ PathVQA (3,600 records)

### Code structure — all modules importable
```python
from medical_rag.models.bioclip import BioCLIPEncoder     # ✅ import OK
from medical_rag.models.biomedbert import BioMedBERTEncoder # ✅ import OK
from medical_rag.models.qwen_vl import QwenVLModel         # ✅ import OK
from medical_rag.models.bge_reranker import BGEReranker    # ✅ import OK
from medical_rag.ingestion.chunker import chunk_text       # ✅ import OK
from medical_rag.ingestion.indexer import build_qdrant_index # ✅ import OK
from medical_rag.agents.graph import AgenticRAGPipeline    # ✅ import OK
```

---

## Requires GPU/API to run fully

These modules import without GPU but **will fail at inference time** without:
- GPU ≥ 16GB VRAM, OR
- Qwen API key in `QWEN_API_KEY` env var

| Module | What fails without GPU |
|---|---|
| `models/bioclip.py` | `encoder.encode_image()` — needs open_clip loaded |
| `models/biomedbert.py` | `encoder.encode()` — needs transformers loaded |
| `models/qwen_vl.py` | `vlm.generate()` — needs torch+transformers or API |
| `models/bge_reranker.py` | `reranker.rerank()` — needs transformers |
| `ingestion/indexer.py` | `build_qdrant_index()` — calls encoders above |
| `agents/*` | All agent nodes that call models |

---

## Requires Docker/Qdrant to run fully

When `use_qdrant=True` in config, the agent nodes call Qdrant.  
Without Docker → **auto-fallback to in-memory Qdrant** (no persistence).

```powershell
# Start Qdrant
docker run -d -p 6333:6333 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

---

## Not yet implemented (Phase 5-7)

| Feature | File | Status |
|---|---|---|
| RAGAS evaluation | `evaluation_advanced.py` | ❌ Not created |
| Exact match / F1 metrics | `evaluation_advanced.py` | ❌ Not created |
| Agent ablation profiles (E/F/G/H) | `ablation.py` | ❌ Not added |
| Gradio demo | `demo/app.py` | ❌ Not created |
| Full benchmark run | `outputs/benchmark/` | ❌ Not run |

---

## Known limitations

1. **Image retrieval is caption-based**: `retrieval/image.py` uses TF-IDF on captions, not actual pixel embeddings. To use BioCLIP you need Qdrant + GPU.
2. **Generation is extractive**: `generation.py` returns formatted text snippets, no LLM reasoning. For VLM you need GPU or API.
3. **Answer accuracy heuristic**: Current `_answer_accuracy()` uses string containment only — not suitable for final paper results.
4. **No data leak check**: Test samples in eval_cases.json may overlap with index — must verify for paper.
