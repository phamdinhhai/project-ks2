# ADR-001: Dùng Qdrant thay vì Chroma hay FAISS

**Date**: 2026-05-30  
**Status**: Accepted

## Context
Cần chọn vector database cho multi-vector index (text 1024-dim + image 512-dim).

## Decision
Dùng **Qdrant** với in-memory fallback khi không có Docker.

## Rationale
- Qdrant hỗ trợ multiple named vectors per collection natively
- Payload filtering: lọc theo `dataset` mà không cần post-filter
- Binary quantization sẵn có → giảm RAM 8x khi cần
- Python client đơn giản, async-ready
- Chroma: không hỗ trợ multi-vector tốt
- FAISS: không có built-in payload filtering, không có server mode

---

# ADR-002: Lazy loading cho tất cả GPU models

**Date**: 2026-05-30  
**Status**: Accepted

## Context
Models (BioCLIP, BioMedBERT, Qwen2.5-VL, BGE) nặng, cần GPU. Baseline pipeline không cần GPU.

## Decision
Import `torch`, `transformers`, `open_clip_torch` **bên trong method**, không ở top-level.

## Rationale
- Baseline chạy được mà không cần cài torch
- Test offline không cần GPU
- Người dùng không có GPU vẫn dùng được baseline
- Tradeoff: lần đầu gọi method sẽ chậm hơn (acceptable)

---

# ADR-003: Dual pipeline — baseline + agent

**Date**: 2026-05-30  
**Status**: Accepted

## Context
Cần cả kết quả ablation baseline (Config A/B/C/D) lẫn agent system cho paper.

## Decision
Giữ `MedicalRAGPipeline` (baseline) và thêm `AgenticRAGPipeline` (agent) song song.

## Rationale
- Baseline là ground truth cho so sánh ablation
- Agent là contribution chính của đồ án
- Không xóa baseline để tránh mất đi điểm so sánh
- Cả hai chia sẻ `RAGConfig` → dễ toggle

---

# ADR-004: 5-patch image chunking

**Date**: 2026-05-30  
**Status**: Accepted

## Context
Cần fine-grained image retrieval — full image không đủ chính xác cho medical domain.

## Decision
Mỗi ảnh tạo 5 patches: `full` + `tl` + `tr` + `bl` + `br` (quadrants).

## Rationale
- VimRAG dùng patch-level để enable region retrieval
- 5 patches = balance giữa granularity và index size
- Quadrants cover all 4 corners where findings typically appear in X-rays
- Crop + re-encode tại query time (stage 2) để không cần lưu trữ hết patches

---

# ADR-005: data/raw_pdfs excluded từ indexing

**Date**: 2026-05-25  
**Status**: Accepted

## Context
`data/raw_pdfs/` chứa PDF sách y khoa tiếng Việt. Cần quyết định có index không.

## Decision
**Exclude** khỏi indexing. Chỉ dùng các dataset tiêu chuẩn (MedQA, BioASQ, ...).

## Rationale
- PDFs cần parser phức tạp (LlamaParse) → out of scope cho phase hiện tại
- Domain của PDFs (Việt Nam) khác với eval benchmarks (English)
- Có thể add lại trong Milestone 8+ sau khi baseline hoàn chỉnh
