# Project Status - 2026-06-07

## Medical Multimodal RAG Agent Status

The Colab evaluation pipeline has completed successfully with a clean final
agent benchmark. The project is now ready for final reporting and optional
larger-scale evaluation.

## Current Stable Components

| Component | Status | Notes |
|---|---|---|
| BioMedBERT text encoder | Stable | Uses `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract` |
| BioCLIP visual encoder | Stable | Uses `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| Qdrant Cloud text collection | Stable | Dense retrieval active |
| Qdrant Cloud image collection | Stable | Visual coarse retrieval active |
| OpenRouter generation | Stable | Gemini 2.5 Flash generation path active |
| Colab workflow | Stable | Drive symlink avoids large `/content` data copies |
| Agent benchmark | Clean | No retrieval fallbacks in latest output |

## Final Clean Output

Imported from:

```text
notebooks/outputs-20260607T075311Z-3-001.zip
```

Local final files:

```text
outputs/benchmark/agent_openrouter.json
outputs/benchmark/agent_openrouter_clean.json
outputs/benchmark/baseline_advanced.json
outputs/benchmark/final_benchmark_summary.md
outputs/ablation/summary.json
outputs/ablation/ablation_report.md
```

## Clean Benchmark Evidence

Latest agent benchmark:

| Metric | Value |
|---|---:|
| Cases | 6 |
| Error count | 0 |
| Qdrant failures | 0 |
| Qdrant dense traces | 6 |
| Visual Stage 1 Qdrant dense traces | 4 |

Resolved failures:

```text
Repo id ... ''
QdrantClient object has no attribute 'search'
BioCLIP dependencies missing
huggingface.co/None
hf-hub:None
Qdrant failed
```

## Ablation Result Summary

| Profile | Recall@5 | MRR@5 | Routing | Image Recall |
|---|---:|---:|---:|---:|
| A | 0.6667 | 0.5000 | 0.6667 | 0.0000 |
| B | 0.6667 | 0.5000 | 0.6667 | 0.0000 |
| C | 1.0000 | 0.8333 | 1.0000 | 1.0000 |
| D | 1.0000 | 0.8333 | 1.0000 | 1.0000 |

## Important Implementation Fixes

1. `evaluate-agent` now passes explicit text and image model names.
2. BioMedBERT/BioCLIP retrievers no longer pass empty model ids.
3. Qdrant retrieval supports both old `search()` and new `query_points()` APIs.
4. BioCLIP encoder falls back to the default model if `None` is provided.
5. Colab 03 explicitly installs and verifies BioCLIP dependencies.

## Known Limitations

- The clean agent benchmark currently uses only 6 eval cases.
- Exact Match and Token F1 are `0.0` because generated answers are citation-rich
  free-form responses, not short exact-answer strings.
- `outputs/` is intentionally ignored by Git; commit only summary reports if
  repository tracking is needed.

## Recommended Next Steps

1. Keep the clean benchmark output as the final small-set validation artifact.
2. Optionally run a larger eval file such as `data/eval_cases_medium_50.json`.
3. Use `outputs/benchmark/final_benchmark_summary.md` in the thesis/report.
4. Commit code/docs only; avoid committing raw outputs unless required.
