# Task List — Medical Multimodal RAG Agent

## Completed in current session

- [x] Build eval set `data/eval_cases_final.json`
  - [x] 275 cases total
  - [x] 162 text cases
  - [x] 113 image cases

- [x] Fix eval Gold ID mismatch
  - [x] Modified `src/medical_rag/eval_case_builder.py`
  - [x] Added canonical source loading from dataset `manifest.json`
  - [x] Fixed ROCO `ROCOv2_*` ID handling
  - [x] Rebuilt `data/eval_cases_final.json`
  - [x] Verified 275/275 gold IDs match indexed IDs/aliases

- [x] Add overlap/data-leak checker
  - [x] Created `scripts/check_data_leak.py`
  - [x] Generated `outputs/data_leak_report.json`

- [x] Debug ROCO/MIMIC zero recall
  - [x] Root cause: eval builder used legacy `data/processed/*.jsonl`, index used canonical manifests
  - [x] ROCO Recall@5 fixed from 0.0000 to 1.0000
  - [x] MIMIC-CXR Recall@5 fixed from 0.0000 to 0.9800

- [x] Run local ablation A/B/C/D after canonical eval fix
  - [x] Output: `outputs/ablation_full/`
  - [x] Report: `outputs/ablation_full/ablation_report.md`

- [x] Run advanced baseline eval EM/F1
  - [x] Output: `outputs/benchmark/baseline_advanced.json`

- [x] Add benchmark summary script
  - [x] Created `scripts/summarize_benchmark.py`
  - [x] Output: `outputs/benchmark/benchmark_summary.md`

- [x] Smoke-test AgenticRAGPipeline locally
  - [x] Installed optional `agent` dependency (`langgraph`)
  - [x] Found performance blocker: agent retrievers reloaded joblib index per case/node
  - [x] Added process-level cached index loading in `agents/text_retriever.py`
  - [x] Added process-level cached index loading in `agents/visual_retriever.py`
  - [x] Created `data/eval_cases_smoke_10.json`
  - [x] Smoke result: `outputs/benchmark/agent_mock_smoke_10.json`
  - [x] Smoke metrics: 10 cases, `error_count=0`, `mean_steps=4.0`

- [x] Add Agent ablation profiles E/F/G/H
  - [x] Added `agent_profile_config()` and `run_agent_ablation()` in `src/medical_rag/ablation.py`
  - [x] Added CLI command `python -m medical_rag ablate-agent`
  - [x] Added forced text-only routing for profile E
  - [x] Added mock/OpenRouter VLM support in agent decomposition
  - [x] Ran smoke subset before full 275-case eval
  - [x] Output: `outputs/agent_ablation_smoke/summary.json`
  - [x] Report: `outputs/agent_ablation_smoke/agent_ablation_report.md`

- [x] Extend benchmark summary with agent ablation
  - [x] Updated `scripts/summarize_benchmark.py` with `--agent-summary`
  - [x] Regenerated `outputs/benchmark/benchmark_summary.md`

## Latest local benchmark results

| Profile | Recall@5 | MRR@5 | Answer Acc | Routing | Image Recall |
|---|---:|---:|---:|---:|---:|
| A text-only | 0.4291 | 0.4054 | 0.6160 | 0.5891 | 0.0000 |
| B text-only + rerank | 0.4291 | 0.4054 | 0.6160 | 0.5891 | 0.0000 |
| C image branch + RRF | 0.8145 | 0.6812 | 0.6160 | 0.7127 | 0.9204 |
| D full baseline | 0.8073 | 0.6781 | 0.6160 | 0.7127 | 0.9204 |

Agent ablation full mock `outputs/agent_ablation_full_mock/summary.json`:

| Profile | EM | Token F1 | Mean Steps | Error Count |
|---|---:|---:|---:|---:|
| E | 0.0000 | 0.0223 | 4.0000 | 0 |
| F | 0.0000 | 0.0223 | 4.0000 | 0 |
| G | 0.0000 | 0.0100 | 4.0000 | 0 |
| H | 0.0000 | 0.0100 | 4.0000 | 0 |

Advanced baseline `outputs/benchmark/baseline_advanced.json`:

| Metric | Value |
|---|---:|
| Recall@5 | 0.8073 |
| MRR@5 | 0.6781 |
| Routing accuracy | 0.7127 |
| Exact Match | 0.0000 |
| Token F1 | 0.0181 |

Per-dataset Recall@5:

| Dataset | Recall@5 | MRR@5 |
|---|---:|---:|
| bioasq | 0.1200 | 0.0497 |
| medqa | 1.0000 | 1.0000 |
| mimic_cxr | 0.9800 | 0.7300 |
| roco | 1.0000 | 1.0000 |
| vqa_rad | 0.8933 | 0.6333 |

## Important local environment note

- `python -m pip install -e ".[agent]"` installed `langgraph 1.2.4`.
- Pip reported dependency conflicts with existing unrelated packages such as `langchain`, `rasa`, and `chromadb`.
- Current project CLI still works for completed baseline/agent smoke tests, but Colab/venv isolation is recommended before larger agent experiments.

## Next implementation tasks

- [x] Run medium/full AgenticRAGPipeline evaluation
  - [x] Created balanced 50-case medium eval subset: `data/eval_cases_medium_50.json`
  - [x] Ran `ablate-agent` on medium subset first
  - [x] Output: `outputs/agent_ablation_medium/summary.json`
  - [x] Report: `outputs/agent_ablation_medium/agent_ablation_report.md`
  - [x] Ran local mock full 275-case agent ablation
  - [x] Output: `outputs/agent_ablation_full_mock/summary.json`
  - [x] Report: `outputs/agent_ablation_full_mock/agent_ablation_report.md`
  - [x] Regenerated combined summary: `outputs/benchmark/benchmark_summary.md`
  - [ ] Cloud/OpenRouter run after secrets configured
  - [ ] Output target: `outputs/benchmark/agent_openrouter.json`

- [x] Add optional RAGAS integration
  - [x] Added `ragas_evaluate()` in `src/medical_rag/evaluation_advanced.py`
  - [x] Added faithfulness metric support
  - [x] Added answer relevancy metric support
  - [x] Added context precision/recall when available
  - [x] Added safe fallback if RAGAS/dependencies/backend unavailable
  - [x] Added CLI flags to `evaluate-advanced`: `--run-ragas`, `--ragas-output-file`, `--ragas-max-samples`
  - [x] Smoke output: `outputs/benchmark/ragas_smoke.json`
  - [x] Smoke status: `available=true`, `skipped=false`, `evaluated_rows=3`

- [ ] Clarify benchmark split semantics
  - [ ] Document retrieval-evidence KB overlap vs model-training data leak
  - [ ] Decide final thesis wording

## Colab / cloud phase

- [ ] Build Qdrant real index
  - [ ] Setup Qdrant Cloud
  - [ ] Setup OpenRouter key
  - [ ] Run small limit test
  - [ ] Run full/semi-full index

- [ ] Run cloud agent evaluation
- [ ] Compare cloud agent vs local lexical baseline

## Demo / report phase

- [ ] Test Gradio demo locally
- [ ] Deploy HuggingFace Spaces
- [ ] Run error analysis
- [ ] Prepare Chapter 4 tables

## Optimization audit — 2026-06-14

- [x] Add free-form answer quality proxy metrics
  - [x] `answer_non_empty_rate`
  - [x] `citation_coverage_rate`
  - [x] `evidence_overlap_rate`
  - [x] `groundedness_proxy_rate`
- [x] Improve benchmark summary notes for thesis/report readiness
- [x] Add offline retrieval error analysis script
- [x] Add lightweight smoke coverage for advanced metrics/error report
- [x] Synchronize README status and result tables
- [ ] Run local verification after edits
- [ ] Optional: run larger OpenRouter/Qdrant benchmark after secrets are configured
- [ ] Optional: clean or ignore untracked notebook copies
