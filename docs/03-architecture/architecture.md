# Architecture — Medical Multimodal RAG Agent

## 4-Layer Architecture

```mermaid
graph TD
    subgraph "LAYER 1 — Data & Knowledge Base"
        D1[MedQA<br>12K docs]
        D2[BioASQ<br>8K docs]
        D3[VQA-RAD<br>2K text+img]
        D4[ROCO<br>12K captions]
        D5[MIMIC-CXR<br>30K reports]
        D6[PathVQA<br>3K pathology]
    end

    subgraph "LAYER 2 — Multimodal Indexing"
        I1[BioMedBERT<br>1024-dim]
        I2[BioCLIP<br>512-dim × 5 patches]
        I3[Qdrant<br>text_chunks + image_patches]
        I4[Joblib fallback<br>BM25 + TF-IDF]
    end

    subgraph "LAYER 3 — LangGraph Agent"
        A1[decompose_query]
        A2[retrieve_text]
        A3[retrieve_visual]
        A4[fuse_and_verify]
        A5[generate_answer]
    end

    subgraph "LAYER 4 — Evaluation"
        E1["Recall@k | MRR | Routing accuracy"]
        E2["RAGAS faithfulness"]
        E3["Ablation A/B/C/D"]
    end

    D1 & D2 & D3 & D4 & D5 & D6 --> I1 & I2
    I1 --> I3
    I2 --> I3
    I1 --> I4
    I3 --> A2 & A3
    I4 --> A2 & A3
    A1 -->|text| A2
    A1 -->|visual| A3
    A1 -->|hybrid| A2 & A3
    A2 & A3 --> A4 --> A5
    A5 --> E1 & E2 & E3
```

## Agent Graph (LangGraph)

```mermaid
graph LR
    START --> decompose
    decompose -->|text_only| text_ret[retrieve_text]
    decompose -->|visual_only| vis_ret[retrieve_visual]
    decompose -->|hybrid| text_ret & vis_ret
    text_ret --> fuse
    vis_ret --> fuse
    fuse[fuse_and_verify] --> gen[generate_answer]
    gen --> END
```

## Dual Pipeline

The project maintains **two pipelines** side by side:

| Pipeline | Class | Index | When to use |
|---|---|---|---|
| Baseline | `MedicalRAGPipeline` | joblib (BM25+TF-IDF) | Ablation, tests, comparison |
| Agent | `AgenticRAGPipeline` | Qdrant (dense multi-vector) | Full system, evaluation target |

Both share `RAGConfig` — toggle with `use_agent`, `use_qdrant`, `use_vlm_generation`.

## Visual Retrieval — 2-Stage Pipeline

```
User image + query
        │
        ▼
  [Stage 1: Coarse]
  BioCLIP encode query (+ blend image 60/40%)
  → Qdrant image_patches search → top-10
        │
        ▼
  [Stage 2: Fine-grained] (optional)
  Qwen2.5-VL ground_region(query, image)
  → crop ROI from each candidate
  → BioCLIP re-encode crops
  → score = 0.5×original + 0.3×confidence + 0.2×similarity
  → top-3 refined results
```

## Data Flow

```
raw JSONL/HF dataset → canonicalize.py → canonical JSONL + manifest.json
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                              data_loaders.py  indexing.py  ingestion/indexer.py
                              (DocumentChunk   (joblib)    (Qdrant)
                               ImageRecord)
```
