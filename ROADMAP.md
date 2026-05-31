# Roadmap — Medical Multimodal RAG Agent

## Completed ✅

### Phase 0: Data Pipeline (Milestone 1)
- [x] 6 datasets canonicalized (MedQA, BioASQ, VQA-RAD, ROCO, MIMIC-CXR, PathVQA)
- [x] Canonical JSONL + manifest.json per dataset
- [x] Data audit CLI (`audit-data`)
- [x] Download + canonicalize scripts

### Phase Baseline: Offline RAG (Milestone 4)
- [x] BM25 + TF-IDF hybrid text retrieval
- [x] Caption-based image retrieval (TF-IDF on captions)
- [x] Rule-based query router (language + modality + dataset detection)
- [x] Late fusion (weighted RRF)
- [x] Lexical overlap reranker
- [x] Extractive answer generator (no LLM)
- [x] Evaluation pipeline (Recall@k, MRR, routing accuracy, error analysis)
- [x] Ablation profiles A/B/C/D
- [x] Static HTML demo export
- [x] CLI commands (12 commands)
- [x] Smoke tests (8 passed)

### Phase 1: Model Wrappers (Milestone 2)
- [x] BioCLIP encoder (512-dim, open_clip_torch)
- [x] BioMedBERT encoder (1024-dim, transformers)
- [x] Qwen2.5-VL wrapper (local 4-bit + API fallback)
- [x] BGE-reranker-v2-m3 cross-encoder

### Phase 2: Qdrant Indexing (Milestone 3)
- [x] Sentence-level text chunker (sliding window)
- [x] 5-patch image chunker (full + 4 quadrants)
- [x] Qdrant multi-vector indexer (auto fallback in-memory)
- [x] Config fields for Qdrant, models, agent modes

### Phase 3: LangGraph Agent (Milestone 5)
- [x] AgentState TypedDict
- [x] Decomposer node (VLM + rule-based fallback)
- [x] Text retriever node (Qdrant + baseline fallback + BGE rerank)
- [x] Visual retriever node (BioCLIP coarse → VLM ROI fine-grained)
- [x] Decision node (fusion + VLM/extractive generation)
- [x] StateGraph assembly with conditional routing

---

## In Progress 🔄

### Phase 5: Advanced Evaluation
- [ ] RAGAS faithfulness + answer relevancy metrics
- [ ] Exact match + F1 answer accuracy
- [ ] Per-dataset recall breakdown
- [ ] Agent-mode ablation profiles (E/F/G/H)

---

## Planned ⏳

### Phase 6: Gradio Demo (Milestone 8)
- [ ] Interactive demo with image upload
- [ ] Evidence visualization + ROI crop display
- [ ] HuggingFace Spaces deployment

### Phase 7: Full Benchmark (Milestone 7)
- [ ] Run 200 VQA-RAD + 100 MedQA + 50 MIMIC-CXR
- [ ] Target: Config D > Config A + 10-15% accuracy on VQA-RAD
- [ ] Error analysis: 20-30 failure cases
- [ ] Inference time benchmark
- [ ] Generate tables for thesis Chapter 4

---

## Priority if short on time

1. Phase 5 (evaluation) — numbers for thesis
2. Phase 7 (benchmark) — results for Chapter 4
3. Phase 6 (demo) — nice to have
