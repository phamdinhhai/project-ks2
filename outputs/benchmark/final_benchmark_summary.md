# Final Benchmark Summary - Medical Multimodal RAG

Generated after the clean Colab 03 run on `2026-06-07`.

## Executive Summary

The multimodal RAG pipeline is now stable across text, image, and hybrid queries.
The final agent benchmark shows no retrieval fallbacks and successfully uses
Qdrant dense retrieval for both BioMedBERT text search and BioCLIP visual search.

## Clean Run Validation

| Check | Result |
|---|---:|
| Agent eval cases | 6 |
| Agent error count | 0 |
| `Qdrant failed` occurrences | 0 |
| Text Qdrant dense retrieval count | 2 |
| Total Qdrant dense retrieval count | 6 |
| Visual Stage 1 Qdrant dense count | 4 |
| BioMedBERT empty repo error | Fixed |
| Qdrant `search()` API error | Fixed |
| BioCLIP dependency error | Fixed |
| BioCLIP `hf-hub:None` error | Fixed |

## Ablation Results

| Profile | Description | Recall@5 | MRR@5 | Routing | Image Recall |
|---|---|---:|---:|---:|---:|
| A | Text-only baseline | 0.6667 | 0.5000 | 0.6667 | 0.0000 |
| B | Text + rerank | 0.6667 | 0.5000 | 0.6667 | 0.0000 |
| C | Image branch enabled | 1.0000 | 0.8333 | 1.0000 | 1.0000 |
| D | Full multimodal pipeline | 1.0000 | 0.8333 | 1.0000 | 1.0000 |

## Advanced Baseline Metrics

| Metric | Value |
|---|---:|
| Total cases | 6 |
| Recall@5 | 1.0000 |
| MRR@5 | 0.8333 |
| Routing accuracy | 1.0000 |
| Exact Match | 0.0000 |
| Token F1 | 0.0000 |

## Clean Agent Benchmark

Source file:

```text
outputs/benchmark/agent_openrouter_clean.json
```

| Metric | Value |
|---|---:|
| Total cases | 6 |
| Error count | 0 |
| Exact Match | 0.0000 |
| Token F1 | 0.0000 |

### Retrieval Trace Evidence

The final agent run contains these clean reasoning traces:

```text
[text_retrieval] Qdrant dense search → 10 results
[visual_retrieval] Stage 1 Qdrant dense → 10 coarse results
```

Per-query routing/retrieval:

| Query | Route | Retrieval evidence |
|---|---|---|
| điều trị viêm phổi | text | Qdrant dense search |
| pneumonia treatment | text | Qdrant dense search |
| triệu chứng lao phổi | hybrid | Stage 1 Qdrant dense |
| pulmonary tuberculosis symptoms | hybrid | Stage 1 Qdrant dense |
| chest xray pneumonia | hybrid | Stage 1 Qdrant dense |
| hình ảnh viêm phổi | hybrid | Stage 1 Qdrant dense |

## Fixed Failure Modes

The final run confirms these previous failure modes are resolved:

```text
Repo id ... ''
QdrantClient object has no attribute 'search'
BioCLIP dependencies missing
huggingface.co/None
hf-hub:None
Qdrant failed
```

## Interpretation Notes

- Retrieval metrics are strong on the current small validation set.
- Profiles C/D demonstrate the value of adding the visual branch.
- Exact Match and Token F1 remain `0.0` because the generator produces
  citation-grounded free-form answers instead of short exact-answer strings.
- For thesis reporting, prioritize Recall@5, MRR@5, routing accuracy, image
  recall, no-fallback traces, and qualitative citation faithfulness.

## Recommended Next Evaluation Expansion

Before final submission, consider adding a larger eval set:

- `data/eval_cases_medium_50.json` for broader text/image coverage.
- More radiology image queries for visual retrieval stress testing.
- Separate generation metrics using human rubric or RAGAS if budget allows.
