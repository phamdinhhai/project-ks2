# Project Status Snapshot — 2026-06-06

## TL;DR

Project hiện đã ổn định local baseline/evaluation trên eval set mới **275 cases**.
Bug lớn ROCO/MIMIC-CXR Recall@5 = 0 đã được sửa bằng cách đồng bộ eval cases với canonical dataset manifests.

Kết quả baseline mới nhất:

| Config | Recall@5 | MRR@5 | Answer Acc | Routing | Image Recall |
|---|---:|---:|---:|---:|---:|
| A text-only | 0.4291 | 0.4054 | 0.6160 | 0.5891 | 0.0000 |
| B text-only + rerank | 0.4291 | 0.4054 | 0.6160 | 0.5891 | 0.0000 |
| C image branch + RRF | **0.8145** | **0.6812** | 0.6160 | 0.7127 | 0.9204 |
| D full baseline | 0.8073 | 0.6781 | 0.6160 | 0.7127 | 0.9204 |

Advanced baseline:

| Metric | Value |
|---|---:|
| Total cases | 275 |
| Recall@5 | 0.8073 |
| MRR@5 | 0.6781 |
| Routing accuracy | 0.7127 |
| Exact Match | 0.0000 |
| Token F1 | 0.0181 |

Agent full mock ablation E/F/G/H đã chạy thành công trên toàn bộ 275-case eval set, `error_count=0` cho toàn bộ profiles.

| Agent Profile | EM | Token F1 | Mean Steps | Error Count |
|---|---:|---:|---:|---:|
| E | 0.0000 | 0.0223 | 4.0000 | 0 |
| F | 0.0000 | 0.0223 | 4.0000 | 0 |
| G | 0.0000 | 0.0100 | 4.0000 | 0 |
| H | 0.0000 | 0.0100 | 4.0000 | 0 |

RAGAS smoke test cũng đã chạy thành công trên 3 evaluated rows:

| Metric | Value |
|---|---:|
| faithfulness | 1.0000 |
| answer_relevancy | 0.3333 |
| context_precision | 0.6667 |
| context_recall | 0.9091 |

---

## Files changed in latest coding session

### Modified

- `src/medical_rag/eval_case_builder.py`
  - Eval builder now prefers canonical dataset `manifest.json` sources.
  - Legacy `data/processed/*.jsonl` is used only as fallback.
  - ROCO `ROCOv2_*` record IDs are preserved to match index IDs.

- `src/medical_rag/agents/text_retriever.py`
  - Added process-level cached index loading for agent fallback retrieval.

- `src/medical_rag/agents/visual_retriever.py`
  - Added process-level cached index loading for agent fallback retrieval.

- `src/medical_rag/agents/decomposer.py`
  - Added `force_text_only_agent` support for agent profile E.
  - Added mock/OpenRouter VLM support in decomposition path.

- `src/medical_rag/ablation.py`
  - Added AgenticRAGPipeline ablation profiles E/F/G/H.
  - Added `run_agent_ablation()` and `agent_profile_config()`.

- `src/medical_rag/__main__.py`
  - Added CLI command `ablate-agent`.

- `TASK.md`
  - Updated current progress and next implementation tasks.

### Added

- `scripts/check_data_leak.py`
  - Checks eval/index overlap and retrieval evidence coverage.

- `scripts/summarize_benchmark.py`
  - Generates thesis-friendly benchmark Markdown summaries.

### Generated

- `data/eval_cases_final.json`
- `data/eval_cases_final_summary.json`
- `data/eval_cases_smoke_10.json`
- `outputs/data_leak_report.json`
- `outputs/ablation_full/`
- `outputs/ablation_full/ablation_report.md`
- `outputs/benchmark/baseline_advanced.json`
- `outputs/benchmark/benchmark_summary.md`
- `outputs/agent_ablation_smoke/summary.json`
- `outputs/agent_ablation_smoke/agent_ablation_report.md`
- `outputs/agent_ablation_medium/summary.json`
- `outputs/agent_ablation_medium/agent_ablation_report.md`
- `outputs/agent_ablation_full_mock/summary.json`
- `outputs/agent_ablation_full_mock/agent_ablation_report.md`
- `outputs/benchmark/ragas_smoke.json`
- `outputs/benchmark/baseline_advanced_smoke_ragas.json`

---

## Important findings

### 1. Canonical eval/index alignment fixed ROCO and MIMIC-CXR

Before fix:

```text
ROCO eval gold:  roco-image-roco-main-0
ROCO index ID:   roco-image-ROCOv2_2023_train_000001

MIMIC eval gold: mimic_cxr-text-mimic_cxr-main-0
MIMIC index ID:  mimic_cxr-text-mimic_cxr-train-0
```

After fix:

```text
ROCO eval gold:  roco-image-ROCOv2_2023_train_000001
MIMIC eval gold: mimic_cxr-text-mimic_cxr-train-0
```

Gold ID coverage is now **275/275**.

### 2. Per-dataset retrieval is now stable except BioASQ

| Dataset | Recall@5 | MRR@5 |
|---|---:|---:|
| MedQA | 1.0000 | 1.0000 |
| VQA-RAD | 0.8933 | 0.6333 |
| ROCO | 1.0000 | 1.0000 |
| MIMIC-CXR | 0.9800 | 0.7300 |
| BioASQ | 0.1200 | 0.0497 |

BioASQ remains the weakest retrieval dataset and should be treated as the next retrieval-quality target if needed.

### 3. Data-leak report is evidence coverage, not model leakage

`outputs/data_leak_report.json` reports overlap between eval gold IDs and index IDs.
For retrieval evaluation, this is expected because gold evidence must exist in the knowledge base.

Thesis wording recommendation: call this **retrieval evidence coverage check**, not model training data leakage.

### 4. Agent local blocker was fixed with index caching

Problem:

- Agent retrievers repeatedly called `joblib.load()` on full index per node/case.
- Full/smoke agent eval appeared to hang.

Fix:

- Added `lru_cache` around fallback `load_indexes()` in text and visual agent retrievers.

Validation:

- Single-case agent run completed.
- 10-case agent smoke eval completed.
- E/F/G/H agent smoke ablation completed with `error_count=0`.
- E/F/G/H agent medium 50-case ablation completed with `error_count=0`.
- E/F/G/H agent full mock 275-case ablation completed with `error_count=0`.

### 6. Optional RAGAS integration now works

Implemented:

- `ragas_evaluate()` in `src/medical_rag/evaluation_advanced.py`.
- CLI flags on `evaluate-advanced`:
  - `--run-ragas`
  - `--ragas-output-file`
  - `--ragas-max-samples`
- Safe fallback when dependencies/backend are unavailable.
- Numeric-only aggregation for RAGAS dataframe outputs.

Smoke validation:

- Output: `outputs/benchmark/ragas_smoke.json`
- `available=true`
- `skipped=false`
- `evaluated_rows=3`

### 7. Local environment warning

`python -m pip install -e ".[agent]"` installed `langgraph 1.2.4` and upgraded `langchain-core`.
Pip reported conflicts with unrelated existing packages such as `langchain`, `rasa`, and `chromadb`.

Recommended for cloud/agent experiments:

- Use a clean Colab runtime or isolated virtual environment.
- Avoid mixing this project with unrelated Rasa/LangChain apps in one Python environment.

---

## Commands already run

```powershell
python -m medical_rag build-eval-cases `
  --data-dir data `
  --output-file data/eval_cases_final.json `
  --summary-file data/eval_cases_final_summary.json `
  --target-count 50

python scripts/check_data_leak.py `
  --eval-file data/eval_cases_final.json `
  --index-dir data/processed/indexes `
  --output outputs/data_leak_report.json

python -m medical_rag evaluate-advanced `
  --eval-file data/eval_cases_final.json `
  --index-dir data/processed/indexes `
  --output-file outputs/benchmark/baseline_advanced.json `
  --top-k 5

python -m medical_rag ablate `
  --eval-file data/eval_cases_final.json `
  --index-dir data/processed/indexes `
  --output-dir outputs/ablation_full `
  --top-k 5

python -m medical_rag summarize-ablation `
  --ablation-dir outputs/ablation_full `
  --output-file outputs/ablation_full/ablation_report.md

python scripts/summarize_benchmark.py `
  --ablation-summary outputs/ablation_full/summary.json `
  --benchmark-file outputs/benchmark/baseline_advanced.json `
  --output-file outputs/benchmark/benchmark_summary.md

python -m medical_rag evaluate-agent `
  --eval-file data/eval_cases_smoke_10.json `
  --index-dir data/processed/indexes `
  --data-dir data `
  --output-file outputs/benchmark/agent_mock_smoke_10.json `
  --top-k 5 `
  --use-mock-models

python -m medical_rag evaluate-advanced `
  --eval-file data/eval_cases_smoke_10.json `
  --index-dir data/processed/indexes `
  --data-dir data `
  --top-k 5 `
  --output-file outputs/benchmark/baseline_advanced_smoke_ragas.json `
  --run-ragas `
  --ragas-output-file outputs/benchmark/ragas_smoke.json `
  --ragas-max-samples 3
```

---

## Recommended next coding task

### Short term

1. Prepare Colab/Qdrant/OpenRouter execution path.
2. Run real OpenRouter/Gemini agent evaluation once secrets are configured.
3. Run larger RAGAS evaluation on generated OpenRouter answers.

### Cloud phase

1. Move to clean Colab runtime.
2. Build Qdrant neural index with BioMedBERT/BioCLIP.
3. Run OpenRouter/Gemini agent evaluation.
4. Compare cloud agent results against local lexical baseline.
