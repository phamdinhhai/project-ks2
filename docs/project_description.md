# ACTION PLAN — Medical Multimodal Fine-grained RAG-based AI Agent
> **Đồ án 2 — Chương trình Kỹ sư**
> Dành cho AI Coding Agent: Đọc file này để hiểu toàn bộ hệ thống cần xây dựng, thứ tự ưu tiên, và các milestone cần kiểm tra.

---

## 0. TÓM TẮT HỆ THỐNG (System Overview)

### Mục tiêu
Xây dựng **Multimodal Fine-grained RAG-based AI Agent** cho domain y tế, có khả năng:
1. Nhận query dạng text + image (X-quang, MRI, pathology slide)
2. Decompose query thành sub-tasks (text retrieval + visual retrieval)
3. Thực hiện fine-grained retrieval: coarse image search → region-of-interest crop → re-rank
4. Fuse evidence từ nhiều nguồn → generate grounded medical answer

### Kiến trúc tổng thể (4 Layers)
```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1 — Data & Knowledge Base                        │
│  VQA-RAD | ROCO | MIMIC-CXR | MedQA | BioASQ | PathVQA │
└────────────────────┬────────────────────────────────────┘
                     │ ingest
┌────────────────────▼────────────────────────────────────┐
│  LAYER 2 — Multimodal Ingestion & Indexing              │
│  Parser → Chunker → Encoder (BioCLIP + BioMedBERT)      │
│  → Qdrant multi-vector index                            │
└────────────────────┬────────────────────────────────────┘
                     │ query
┌────────────────────▼────────────────────────────────────┐
│  LAYER 3 — Agentic Fine-grained RAG (LangGraph)         │
│  Decomposition Agent                                    │
│    ├── Text Retriever (dense + BM25 hybrid)             │
│    └── Visual Retriever (BioCLIP coarse → ROI crop)     │
│  Decision Agent (fusion + verification)                 │
│  Generator (Qwen2.5-VL-7B)                              │
└────────────────────┬────────────────────────────────────┘
                     │ evaluate
┌────────────────────▼────────────────────────────────────┐
│  LAYER 4 — Evaluation Pipeline                          │
│  RAGAS | Accuracy | Faithfulness | Hit Rate | #Steps    │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack (pinned versions)
| Component | Library | Version |
|---|---|---|
| Agent framework | LangGraph | 0.2.28 |
| RAG framework | LlamaIndex | 0.11.x |
| Vector DB | Qdrant | latest stable |
| Image embedding | BioCLIP | (reuse existing) |
| Text embedding | BioMedBERT-large | HuggingFace |
| VLM | Qwen2.5-VL-7B-Instruct | 4-bit quant |
| Reranker | BGE-reranker-v2-m3 | HuggingFace |
| Parser | LlamaParse + Pillow | latest |
| Evaluation | RAGAS | 0.2.x |
| Demo | Gradio | latest |

---

## 1. CẤU TRÚC CODEBASE

```
medical-multimodal-rag-agent/
├── data/
│   ├── bioasq/
│   │   └── hf_dataset/
│   ├── medqa/
│   │   ├── medqa_test.jsonl
│   │   ├── medqa_train.jsonl
│   │   └── medqa_validation.jsonl
│   ├── mimic_cxr/
│   │   ├── hf_dataset/
│   │   ├── images/
│   │   ├── raw_pdfs/
│   │   └── processed/           ← output của ingestion pipeline
│   ├── roco/
│   │   ├── hf_subset_2_5gb/
│   │   └── roco_subset_2_5gb.jsonl
│   ├── vqa_rad/
│   │   ├── hf_dataset/
│   │   ├── images/
│   │   └── vqa_rad_train.jsonl
│   └── pathvqa/                  ← tải thêm từ HuggingFace
│       └── hf_dataset/
│
├── src/
│   ├── __init__.py
│   ├── config.py                 ← tất cả config tập trung đây
│   │
│   ├── support_repo/             ← tài liệu tham khảo, KHÔNG import trực tiếp vào production code
│   │   ├── HMRAG_project/
│   │   │   ├── HMRAG/            ← source repo đã clone
│   │   │   └── 2504.12330v1.md   ← paper (đọc file .md này, KHÔNG đọc PDF)
│   │   ├── VimRAG_project/
│   │   │   ├── VRAG/             ← source repo đã clone
│   │   │   └── 2602.12735v2.md   ← paper (đọc file .md này, KHÔNG đọc PDF)
│   │   ├── MMedRAG_project/
│   │   │   ├── MMed-RAG/         ← source repo đã clone
│   │   │   └── 2410.13085v2.md   ← paper (đọc file .md này, KHÔNG đọc PDF)
│   │   └── A-MAR_project/
│   │       ├── A-MAR/            ← source repo đã clone
│   │       └── 2604.19689v1.md   ← paper (đọc file .md này, KHÔNG đọc PDF)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audit.py              ← kiểm tra schema từng dataset
│   │   ├── normalizer.py         ← chuẩn hóa về UnifiedSample format
│   │   └── loader.py             ← DataLoader cho từng dataset
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py             ← LlamaParse (PDF) + image loader
│   │   ├── chunker.py            ← fine-grained chunking strategy
│   │   └── indexer.py            ← Qdrant multi-vector indexing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bioclip.py            ← BioCLIP image encoder (reuse)
│   │   ├── biomedbert.py         ← BioMedBERT text encoder
│   │   └── qwen_vl.py            ← Qwen2.5-VL wrapper
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py              ← LangGraph AgentState TypedDict
│   │   ├── decomposer.py         ← QueryDecompositionAgent node
│   │   ├── text_retriever.py     ← dense + BM25 hybrid retrieval
│   │   ├── visual_retriever.py   ← BioCLIP coarse → ROI crop fine
│   │   ├── decision.py           ← FusionVerifier + AnswerGenerator
│   │   └── graph.py              ← LangGraph StateGraph assembly
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py            ← RAGAS + custom medical metrics
│       └── benchmark.py          ← full eval loop, save JSON results
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_rag.ipynb
│   ├── 03_agent_dev.ipynb
│   └── 04_ablation_study.ipynb
│
├── demo/
│   └── app.py                    ← Gradio demo
│
├── experiments/
│   └── results/
│       ├── config_A_baseline.json
│       ├── config_B_agent_only.json
│       ├── config_C_finegrained_only.json
│       └── config_D_proposed_full.json
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_agent.py
│
├── requirements.txt
├── README.md
└── ACTIONPLAN.md                 ← file này
```

---

## 2. SUPPORT REPOS — TÀI LIỆU THAM KHẢO CHO AI CODING AGENT

> **Hướng dẫn cho AI:** Trước khi implement bất kỳ module nào, hãy đọc các file paper `.md` và duyệt source code của repo tương ứng để hiểu ý tưởng cốt lõi. Mục tiêu là **tái sử dụng logic đã được chứng minh**, không phải copy nguyên xi. Tất cả các file paper đều ở dạng `.md` — KHÔNG đọc file `.pdf`.

---

### 2.1 HM-RAG — Hierarchical Multi-Agent RAG
**Paper:** `src/support_repo/HMRAG_project/2504.12330v1.md`
**Repo:** `src/support_repo/HMRAG_project/HMRAG/`
**arXiv:** 2504.12330 | **Venue:** ACM MM 2025
**GitHub gốc:** https://github.com/ocean-luna/HMRAG

#### Kiến thức cốt lõi cần nắm (đọc paper để hiểu):
- **3-tier hierarchical agent**: Decomposition Agent → Multi-source Retrieval Agent → Decision Agent
- **Query decomposition strategy**: cách phân tách 1 complex query thành các atomic sub-queries
- **Multi-source retrieval**: cách orchestrate nhiều retrieval sources song song
- **Decision agent**: cách merge evidence từ nhiều nguồn và resolve conflicts
- **Evaluation protocol trên ScienceQA**: metric setup, baseline comparison

#### Những gì cần đọc trong repo:
```
HMRAG/
├── agents/          ← đọc để hiểu LangGraph node structure
├── retrieval/       ← đọc để hiểu multi-source retrieval logic
├── evaluation/      ← đọc để reuse evaluation script structure
└── README.md        ← đọc để hiểu cách chạy và config
```

#### Reuse strategy cho project này:
| Component HM-RAG | Reuse trong project | Cách adapt |
|---|---|---|
| `DecompositionAgent` | `src/agents/decomposer.py` | Thêm medical query patterns, detect image presence |
| `MultiSourceRetriever` | `src/agents/text_retriever.py` | Thay dataset source → Qdrant medical collections |
| `DecisionAgent` | `src/agents/decision.py` | Thêm medical faithfulness check |
| Evaluation script | `src/evaluation/benchmark.py` | Adapt metrics cho medical domain |
| LangGraph graph structure | `src/agents/graph.py` | Thêm visual retrieval branch |

#### Điểm khác biệt project này so với HM-RAG:
- HM-RAG: text-only, ScienceQA domain
- Project này: multimodal (text + image), medical domain, thêm visual fine-grained retrieval stage

---

### 2.2 VimRAG / VRAG — Progressive Visual Retrieval
**Paper:** `src/support_repo/VimRAG_project/2602.12735v2.md`
**Repo:** `src/support_repo/VimRAG_project/VRAG/`
**arXiv:** 2602.12735
**GitHub gốc:** https://github.com/Alibaba-NLP/VRAG

#### Kiến thức cốt lõi cần nắm (đọc paper để hiểu):
- **Multimodal Memory Graph**: cách build graph kết nối image patches với text context
- **Graph-Modulated Visual Memory Encoding**: cách allocate visual tokens dựa trên relevance
- **Progressive retrieval**: coarse → fine, từ document-level đến patch-level
- **Fine-grained token allocation**: không dùng uniform số lượng token cho mọi ảnh — ảnh quan trọng hơn được cấp nhiều token hơn
- **DashScope API integration**: cách demo chạy không cần GPU mạnh

#### Những gì cần đọc trong repo:
```
VRAG/
├── visual_memory/   ← đọc để hiểu patch encoding + memory graph
├── retrieval/       ← đọc để hiểu 2-stage coarse-to-fine logic
├── demo/            ← đọc để hiểu cách integrate với VLM API
└── README.md        ← đọc để hiểu setup và inference pipeline
```

#### Reuse strategy cho project này:
| Component VimRAG | Reuse trong project | Cách adapt |
|---|---|---|
| 2-stage coarse→fine retrieval | `src/agents/visual_retriever.py` (Stage 1 + Stage 2) | Thay general images → medical images (X-ray, MRI) |
| Patch extraction logic | `src/ingestion/chunker.py` hàm `chunk_image()` | Dùng 5-patch strategy (full + 4 quadrants) |
| Token allocation strategy | `src/agents/visual_retriever.py` scoring | Adapt cho BioCLIP scoring thay vì CLIP gốc |
| Visual grounding | `src/models/qwen_vl.py` hàm `ground_region()` | Dùng Qwen2.5-VL thay model gốc của VRAG |

#### Điểm khác biệt project này so với VimRAG:
- VimRAG: general visual documents, không chuyên medical
- Project này: medical-specific (radiology + pathology), dùng BioCLIP thay CLIP thông thường → better domain alignment

---

### 2.3 MMed-RAG — Multimodal Medical RAG ⭐ (Đọc trước tiên)
**Paper:** `src/support_repo/MMedRAG_project/2410.13085v2.md`
**Repo:** `src/support_repo/MMedRAG_project/MMed-RAG/`
**arXiv:** 2410.13085 | **Venue:** ICLR 2025
**GitHub gốc:** https://github.com/richard-peng-xia/MMed-RAG

> ⭐ **Đây là paper gần nhất với đề tài.** Đọc paper này TRƯỚC để hiểu state-of-the-art trong medical multimodal RAG, từ đó xác định rõ gap mà project này lấp đầy.

#### Kiến thức cốt lõi cần nắm (đọc paper để hiểu):
- **Adaptive context selection**: cách chọn số lượng retrieved context phù hợp cho từng query
- **Medical multi-modal alignment**: cách align text và image trong medical domain
- **Diverse and compatible evidence**: cách đảm bảo retrieved evidence không redundant
- **Domain-specific retrieval**: pipeline chuyên cho radiology và pathology
- **Benchmark setup trên VQA-RAD, ROCO, MIMIC**: đây chính là dataset bạn đang dùng → so sánh số liệu trực tiếp

#### Những gì cần đọc trong repo:
```
MMed-RAG/
├── retrieval/       ← đọc để hiểu medical-specific retrieval strategy
├── models/          ← đọc để xem model architecture và embedding approach
├── evaluation/      ← QUAN TRỌNG: reuse eval script cho VQA-RAD (cùng dataset)
├── data/            ← đọc để hiểu cách họ process VQA-RAD, ROCO, MIMIC-CXR
└── README.md        ← đọc để reproduce baseline của họ
```

#### Reuse strategy cho project này:
| Component MMed-RAG | Reuse trong project | Cách adapt |
|---|---|---|
| Medical retrieval pipeline | Tham khảo cho `src/ingestion/indexer.py` | Thêm Qdrant multi-vector thay FAISS |
| VQA-RAD evaluation protocol | `src/evaluation/metrics.py` | Dùng cùng split + metric để so sánh trực tiếp |
| MIMIC-CXR processing | `src/data/normalizer.py` hàm `normalize_mimic_cxr()` | Tham khảo cách extract question từ report |
| Adaptive context selection | `src/agents/decision.py` | Thêm dynamic context sizing |

#### Điểm khác biệt project này so với MMed-RAG:
- MMed-RAG: document-level retrieval, không có agent, không có fine-grained visual
- Project này: **thêm hierarchical multi-agent** (từ HM-RAG) + **fine-grained visual region retrieval** (từ VimRAG) → đây là contribution chính

#### ⚠️ Quan trọng cho báo cáo:
Số liệu của MMed-RAG trên VQA-RAD là **baseline chính để so sánh** trong Chương 4. Project này phải đạt accuracy cao hơn MMed-RAG trên cùng test split.

---

### 2.4 A-MAR — Agent-based Multimodal Adaptive Retrieval
**Paper:** `src/support_repo/A-MAR_project/2604.19689v1.md`
**Repo:** `src/support_repo/A-MAR_project/A-MAR/`
**arXiv:** 2604.19689
**GitHub gốc:** https://github.com/ShuaiWang97/A-MAR

#### Kiến thức cốt lõi cần nắm (đọc paper để hiểu):
- **Structured reasoning plan**: cách agent lập kế hoạch trước khi retrieve
- **Adaptive retrieval**: cách quyết định khi nào cần retrieve thêm, khi nào đủ
- **Fine-grained structured output**: cách format reasoning steps thành structured JSON
- **Agent stopping criteria**: cách xác định khi nào agent nên dừng lặp

#### Những gì cần đọc trong repo:
```
A-MAR/
├── agent/           ← đọc để hiểu reasoning plan + adaptive loop
├── retrieval/       ← đọc để hiểu structured retrieval với plan
└── README.md        ← đọc để hiểu pipeline tổng thể
```

#### Reuse strategy cho project này:
| Component A-MAR | Reuse trong project | Cách adapt |
|---|---|---|
| Structured reasoning plan | `src/agents/decomposer.py` | Thêm explicit plan step trước decomposition |
| Adaptive retrieval loop | `src/agents/graph.py` conditional routing | Thêm "need more evidence?" check node |
| Stopping criteria | `src/agents/decision.py` | Thêm confidence threshold check |

#### Điểm khác biệt project này so với A-MAR:
- A-MAR: general domain, không medical-specific
- Project này: medical domain + BioCLIP visual encoding + hierarchical agent từ HM-RAG

---

### 2.5 Tổng hợp: Mapping Paper → Module trong Project

```
Project Module                    ← Lấy ý tưởng từ
─────────────────────────────────────────────────────
src/agents/decomposer.py          ← HM-RAG (Decomposition Agent)
                                     A-MAR (Structured reasoning plan)
src/agents/text_retriever.py      ← HM-RAG (Multi-source retrieval)
                                     MMed-RAG (Medical text retrieval)
src/agents/visual_retriever.py    ← VimRAG (2-stage coarse→fine)
                                     A-MAR (Adaptive retrieval)
src/agents/decision.py            ← HM-RAG (Decision Agent)
                                     MMed-RAG (Adaptive context selection)
                                     A-MAR (Stopping criteria)
src/agents/graph.py               ← HM-RAG (LangGraph structure)
                                     A-MAR (Adaptive loop)
src/ingestion/chunker.py          ← VimRAG (Patch extraction)
src/models/bioclip.py             ← MMed-RAG (Medical image embedding)
src/evaluation/metrics.py         ← MMed-RAG (VQA-RAD eval protocol)
src/evaluation/benchmark.py       ← HM-RAG (Evaluation script structure)
```

---

### ✅ MILESTONE PRE-0 — Đọc & Duyệt Support Repos
**Deadline: Ngày đầu tiên, trước khi viết bất kỳ dòng code nào**

**Checklist đọc paper (theo thứ tự ưu tiên):**
- [ ] Đọc `src/support_repo/MMedRAG_project/2410.13085v2.md` — ghi chú số liệu VQA-RAD accuracy của họ
- [ ] Đọc `src/support_repo/HMRAG_project/2504.12330v1.md` — ghi chú LangGraph graph structure
- [ ] Đọc `src/support_repo/VimRAG_project/2602.12735v2.md` — ghi chú 2-stage visual retrieval flow
- [ ] Đọc `src/support_repo/A-MAR_project/2604.19689v1.md` — ghi chú adaptive loop + stopping criteria

**Checklist duyệt repo (theo thứ tự ưu tiên):**
- [ ] Duyệt `src/support_repo/MMedRAG_project/MMed-RAG/` — tìm evaluation script cho VQA-RAD
- [ ] Duyệt `src/support_repo/HMRAG_project/HMRAG/` — tìm LangGraph agent node definitions
- [ ] Duyệt `src/support_repo/VimRAG_project/VRAG/` — tìm patch extraction + coarse-to-fine logic
- [ ] Duyệt `src/support_repo/A-MAR_project/A-MAR/` — tìm adaptive retrieval loop

**Checklist ghi chú sau khi đọc:**
- [ ] Ghi số liệu baseline của MMed-RAG trên VQA-RAD vào `experiments/results/mmedrag_baseline_reference.json`
- [ ] Ghi các hàm/class có thể reuse từ mỗi repo vào phần comment đầu file tương ứng trong `src/`
- [ ] Xác nhận: project này khác gì so với 4 papers trên (ghi vào `docs/project_description.md` phần "Contribution")

**Review checklist khi hoàn thành:**
- [ ] Có thể trả lời: "MMed-RAG đạt bao nhiêu % accuracy trên VQA-RAD test split?"
- [ ] Có thể trả lời: "HM-RAG dùng bao nhiêu agent nodes trong LangGraph?"
- [ ] Có thể trả lời: "VimRAG dùng bao nhiêu stages trong visual retrieval?"
- [ ] Có thể trả lời: "A-MAR dừng adaptive loop khi nào?"

---

## 3. DATA SCHEMA CHUẨN (UnifiedSample)

Mọi dataset sau khi qua `normalizer.py` đều về format này:

```python
# src/data/normalizer.py
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class UnifiedSample:
    id: str                          # "{dataset}_{original_id}"
    dataset: str                     # "vqa_rad" | "roco" | "mimic_cxr" | "medqa" | "bioasq" | "pathvqa"
    question: str                    # câu hỏi hoặc query
    answer: str                      # ground truth answer
    image_path: Optional[str]        # None nếu text-only
    modality: Optional[str]          # "chest_xray" | "mri" | "pathology" | "ct" | None
    split: str                       # "train" | "test" | "validation"
    metadata: dict                   # raw fields gốc của từng dataset
```

### Schema mapping từng dataset

| Dataset | question field | answer field | image field | modality |
|---|---|---|---|---|
| VQA-RAD | `question` | `answer` | `image_name` → `data/vqa_rad/images/` | detect từ metadata |
| ROCO | `caption` (dùng làm question) | N/A (retrieval task) | `file_name` | mixed |
| MIMIC-CXR | extract từ report section | impression section | `dicom_id` | chest_xray |
| MedQA | `question` | `answer` (trong `options`) | None | None |
| BioASQ | `body` | `ideal_answer[0]` | None | None |
| PathVQA | `question` | `answer` | `image` | pathology |

---

## 4. CÁC MODULE CẦN XÂY DỰNG — CHI TIẾT

### MODULE A: Data Audit & Normalizer

**File:** `src/data/audit.py`

Chức năng:
- In ra schema của từng JSONL/HF dataset
- Đếm số sample, kiểm tra missing fields
- Phát hiện image paths bị thiếu

```python
# Interface cần implement
def audit_dataset(dataset_name: str, data_path: str) -> dict:
    """
    Returns:
        {
            "total_samples": int,
            "fields": list[str],
            "missing_image_count": int,
            "sample_example": dict
        }
    """
```

**File:** `src/data/normalizer.py`

```python
# Interface cần implement
def normalize_vqa_rad(raw_path: str) -> list[UnifiedSample]: ...
def normalize_roco(raw_path: str) -> list[UnifiedSample]: ...
def normalize_mimic_cxr(raw_path: str) -> list[UnifiedSample]: ...
def normalize_medqa(raw_path: str) -> list[UnifiedSample]: ...
def normalize_bioasq(raw_path: str) -> list[UnifiedSample]: ...
def normalize_pathvqa(raw_path: str) -> list[UnifiedSample]: ...
def normalize_all(data_root: str) -> list[UnifiedSample]: ...  # gọi tất cả
```

---

### MODULE B: Ingestion Pipeline

**File:** `src/ingestion/chunker.py`

Fine-grained chunking strategy:
- Text: sentence-level split (spaCy hoặc NLTK sent_tokenize) với sliding window overlap 1 sentence
- Image: chia thành 4 patches (top-left, top-right, bottom-left, bottom-right) + full image → 5 representations mỗi ảnh

```python
# Interface
def chunk_text(text: str, window_size: int = 3, overlap: int = 1) -> list[str]:
    """Trả về list các text chunks ở cấp sentence"""

def chunk_image(image_path: str) -> list[dict]:
    """
    Returns list of:
    {
        "patch_id": "full" | "tl" | "tr" | "bl" | "br",
        "image": PIL.Image,
        "bbox": (x1, y1, x2, y2)  # relative coords 0-1
    }
    """
```

**File:** `src/ingestion/indexer.py`

Qdrant collection schema:
- Collection `text_chunks`: vector từ BioMedBERT (768-dim), payload gồm {text, sample_id, dataset, chunk_idx}
- Collection `image_patches`: vector từ BioCLIP (512-dim), payload gồm {image_path, patch_id, sample_id, dataset, caption}

```python
# Interface
def init_qdrant_collections(client: QdrantClient) -> None:
    """Tạo 2 collections nếu chưa có"""

def index_text_chunks(client: QdrantClient, chunks: list[dict]) -> int:
    """Returns số chunks đã index"""

def index_image_patches(client: QdrantClient, patches: list[dict]) -> int:
    """Returns số patches đã index"""
```

---

### MODULE C: Models

**File:** `src/models/bioclip.py` (reuse từ thesis, wrap lại)

```python
class BioCLIPEncoder:
    def encode_image(self, image: PIL.Image) -> np.ndarray:  # (512,)
    def encode_text(self, text: str) -> np.ndarray:          # (512,)
    def encode_batch(self, images: list) -> np.ndarray:      # (N, 512)
```

**File:** `src/models/biomedbert.py`

```python
class BioMedBERTEncoder:
    # Model: "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract-fulltext"
    def encode(self, text: str) -> np.ndarray:               # (1024,)
    def encode_batch(self, texts: list[str]) -> np.ndarray:  # (N, 1024)
```

**File:** `src/models/qwen_vl.py`

```python
class QwenVLModel:
    # Model: "Qwen/Qwen2.5-VL-7B-Instruct" với 4-bit quantization
    def generate(self, question: str, image: Optional[PIL.Image], context: str) -> str:
    def ground_region(self, question: str, image: PIL.Image) -> dict:
        """
        Returns: {"bbox": [x1,y1,x2,y2], "confidence": float, "description": str}
        Dùng cho visual region proposal trong fine-grained retrieval
        """
```

---

### MODULE D: Agent System (LangGraph)

**File:** `src/agents/state.py`

```python
from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Input
    question: str
    image_path: Optional[str]
    
    # Decomposition output
    text_subquery: str
    visual_subquery: str
    query_type: str  # "text" | "visual" | "hybrid"
    
    # Retrieval results
    text_evidence: list[dict]    # [{text, score, source}, ...]
    visual_evidence: list[dict]  # [{image_path, patch_id, caption, score}, ...]
    
    # Decision output
    fused_evidence: str          # formatted context string
    faithfulness_score: float
    
    # Final output
    answer: str
    reasoning_steps: list[str]
    citations: list[str]
    
    # Agent metadata
    step_count: int
    error: Optional[str]
```

**File:** `src/agents/decomposer.py`

```python
def decompose_query(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Logic:
    1. Nếu có image → query_type = "hybrid" hoặc "visual"
    2. Nếu text-only → query_type = "text"
    3. Dùng Qwen2.5-VL phân tích question để extract:
       - text_subquery: phần cần tìm trong text knowledge base
       - visual_subquery: phần cần tìm trong image knowledge base
    
    Prompt template cho decomposition:
    "Given medical question: {question}
     Decompose into:
     1. Text sub-query (for searching medical text documents)
     2. Visual sub-query (for searching medical images)
     Return JSON: {text_subquery: str, visual_subquery: str}"
    """
```

**File:** `src/agents/text_retriever.py`

```python
def retrieve_text(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Pipeline:
    1. BioMedBERT encode text_subquery → dense vector
    2. Qdrant search collection "text_chunks" → top-20 by cosine
    3. BM25 score các kết quả → hybrid score = 0.7*dense + 0.3*bm25
    4. BGE-reranker cross-encode top-20 → giữ top-5
    5. Update state.text_evidence
    """
```

**File:** `src/agents/visual_retriever.py`

```python
def retrieve_visual(state: AgentState) -> AgentState:
    """
    LangGraph node — 2 stages:
    
    Stage 1 (Coarse):
    - BioCLIP encode visual_subquery (text) → query vector
    - Qdrant search collection "image_patches" → top-10 full images
    
    Stage 2 (Fine-grained):
    - Nếu state.image_path tồn tại: dùng Qwen2.5-VL ground region
      → crop ROI từ top-10 images dựa trên visual similarity
    - Qwen2.5-VL generate caption cho mỗi cropped region
    - Score lại dựa trên caption relevance
    - Giữ top-3 regions
    
    Update state.visual_evidence
    """
```

**File:** `src/agents/decision.py`

```python
def fuse_and_verify(state: AgentState) -> AgentState:
    """
    LangGraph node.
    1. Format text_evidence + visual_evidence thành unified context string
    2. Check consistency: nếu text và visual evidence mâu thuẫn → flag
    3. Update state.fused_evidence, state.faithfulness_score
    """

def generate_answer(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Prompt Qwen2.5-VL với:
    - Original question
    - Original image (nếu có)
    - Fused evidence context
    → Generate answer với citations
    → Update state.answer, state.citations, state.reasoning_steps
    """
```

**File:** `src/agents/graph.py`

```python
from langgraph.graph import StateGraph, END

def build_agent_graph() -> StateGraph:
    """
    Graph flow:
    
    decompose_query
         │
         ├─ (query_type == "text") ──→ retrieve_text ──────────────→ fuse_and_verify
         │                                                                    │
         ├─ (query_type == "visual") ─→ retrieve_visual ─────────────→ fuse_and_verify
         │                                                                    │
         └─ (query_type == "hybrid") ─→ retrieve_text ──┐                   │
                                        retrieve_visual ─┘→ fuse_and_verify ─┘
                                                                             │
                                                                    generate_answer
                                                                             │
                                                                           END
    """
```

---

### MODULE E: Evaluation

**File:** `src/evaluation/metrics.py`

Metrics cần implement:
```python
def accuracy_exact_match(pred: str, gold: str) -> float:
    """Exact match sau khi normalize (lowercase, strip punctuation)"""

def accuracy_llm_judge(pred: str, gold: str, question: str, model) -> float:
    """Dùng Qwen2.5-VL-3B để judge: score 0/0.5/1.0"""

def retrieval_hit_rate(retrieved_ids: list, gold_ids: list, k: int) -> float:
    """Hit Rate@K: có ít nhất 1 gold doc trong top-K không"""

def retrieval_recall_at_k(retrieved_ids: list, gold_ids: list, k: int) -> float:
    """Recall@K"""

def ragas_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Dùng RAGAS 0.2+ faithfulness metric"""

def ragas_answer_relevancy(question: str, answer: str) -> float:
    """RAGAS answer relevancy"""
```

**File:** `src/evaluation/benchmark.py`

```python
def run_benchmark(
    agent_graph,              # LangGraph compiled graph
    eval_samples: list,       # list UnifiedSample
    config_name: str,         # "A_baseline" | "B_agent_only" | "C_finegrained" | "D_proposed"
    output_path: str
) -> dict:
    """
    Chạy từng sample, ghi kết quả vào experiments/results/{config_name}.json
    Returns summary metrics dict
    """
```

---

## 5. ABLATION STUDY — 4 CẤU HÌNH

| Config | Agent | Fine-grained | Mô tả |
|---|---|---|---|
| **A — Baseline** | ✗ | ✗ | Vanilla RAG: LlamaIndex + single retrieval + direct generate |
| **B — Agent only** | ✓ | ✗ | Multi-agent (decompose + fuse) nhưng standard retrieval |
| **C — Fine-grained only** | ✗ | ✓ | Progressive visual retrieval nhưng không có agent |
| **D — Proposed (Full)** | ✓ | ✓ | Toàn bộ hệ thống |

Eval set:
- **200 samples VQA-RAD** (visual QA)
- **100 samples MedQA** (text QA)
- **50 samples MIMIC-CXR** (report-based QA, tự tạo)

Target: Config D đạt accuracy cao hơn Config A ít nhất **+10–15%** trên VQA-RAD.

---

## 6. MILESTONES & CHECKLIST

---

### ✅ MILESTONE 0 — Environment Setup
**Deadline: Tuần 1, Ngày 1–2**

**Setup checklist:**
- [ ] Tạo GitHub repo `medical-multimodal-rag-agent`
- [ ] Fork `https://github.com/ocean-luna/HMRAG` vào repo mới
- [ ] Tạo conda environment `medrag` với Python 3.11
- [ ] Cài đặt tất cả dependencies trong `requirements.txt`
- [ ] Pin `langgraph==0.2.28`
- [ ] Pin `ragas==0.2.x`
- [ ] Setup Qdrant bằng Docker: `docker run -p 6333:6333 qdrant/qdrant`
- [ ] Test Qdrant kết nối: `curl http://localhost:6333/health` trả về `{"status":"ok"}`
- [ ] Chạy thử HM-RAG gốc trên ScienceQA dataset của repo — pipeline end-to-end chạy được

**Review checklist khi hoàn thành:**
- [ ] `python -c "import langgraph; print(langgraph.__version__)"` in ra đúng version
- [ ] `python -c "import ragas; print(ragas.__version__)"` in ra đúng version
- [ ] Qdrant health check pass
- [ ] HM-RAG gốc chạy được ít nhất 10 samples không lỗi
- [ ] Commit đầu tiên lên GitHub với message "feat: initial setup"

---

### ✅ MILESTONE 1 — Data Audit & Normalization
**Deadline: Tuần 1, Ngày 3–5**

**Build checklist:**
- [ ] Viết `src/data/audit.py` — hàm `audit_dataset()` cho từng dataset
- [ ] Chạy audit và ghi kết quả vào `notebooks/01_data_exploration.ipynb`
- [ ] Xác nhận schema thực tế của từng file JSONL/HF dataset
- [ ] Viết `src/data/normalizer.py` với 6 hàm normalize (vqa_rad, roco, mimic_cxr, medqa, bioasq, pathvqa)
- [ ] Tải PathVQA: `datasets.load_dataset("flaviagiammarino/path-vqa")` → lưu vào `data/pathvqa/`
- [ ] Viết `src/data/loader.py` — DataLoader wrapper

**Review checklist khi hoàn thành:**
- [ ] `normalize_vqa_rad()` trả về list `UnifiedSample`, kiểm tra 5 samples có đúng fields
- [ ] `normalize_roco()` trả về list `UnifiedSample` với image_path hợp lệ (file tồn tại)
- [ ] `normalize_medqa()` trả về list với `image_path=None`
- [ ] `normalize_all()` chạy không lỗi, in tổng số samples mỗi dataset
- [ ] Tổng số samples sau normalize: ghi vào `README.md` phần "Dataset Statistics"
- [ ] Commit: "feat: data normalization pipeline"

---

### ✅ MILESTONE 2 — Models & Encoders
**Deadline: Tuần 2, Ngày 1–3**

**Build checklist:**
- [ ] Wrap BioCLIP từ thesis vào `src/models/bioclip.py` với interface chuẩn
- [ ] Implement `src/models/biomedbert.py` — load `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract-fulltext`
- [ ] Implement `src/models/qwen_vl.py` — load Qwen2.5-VL-7B-Instruct với 4-bit quantization (bitsandbytes)
- [ ] Test encode 1 image qua BioCLIP → vector shape (512,)
- [ ] Test encode 1 text qua BioMedBERT → vector shape (1024,)
- [ ] Test Qwen2.5-VL generate text từ image + question

**Review checklist khi hoàn thành:**
- [ ] BioCLIP: `encode_image(img).shape == (512,)` ✓
- [ ] BioMedBERT: `encode("chest pain").shape == (1024,)` ✓
- [ ] Qwen2.5-VL: generate answer cho 1 VQA-RAD sample — output hợp lý (không hallucinate hoàn toàn)
- [ ] GPU memory usage < 16GB khi load đồng thời BioCLIP + Qwen2.5-VL-7B-4bit
- [ ] Ghi benchmark inference time: BioCLIP /image, BioMedBERT /text, Qwen2.5-VL /query
- [ ] Commit: "feat: model encoders"

---

### ✅ MILESTONE 3 — Ingestion & Indexing
**Deadline: Tuần 2, Ngày 4–7**

**Build checklist:**
- [ ] Implement `src/ingestion/chunker.py`:
  - `chunk_text()` — sentence-level với sliding window
  - `chunk_image()` — 5 patches (full + 4 quadrants)
- [ ] Implement `src/ingestion/indexer.py`:
  - `init_qdrant_collections()` — tạo 2 collections với đúng vector dimensions
  - `index_text_chunks()` — batch index text vectors vào Qdrant
  - `index_image_patches()` — batch index image vectors vào Qdrant
- [ ] Implement `src/ingestion/parser.py`:
  - Parse MIMIC-CXR PDF reports → text + extract images
  - Load images từ VQA-RAD, ROCO
- [ ] Chạy full ingestion pipeline trên **subset nhỏ để test**: 100 VQA-RAD + 100 ROCO

**Review checklist khi hoàn thành:**
- [ ] Qdrant collection `text_chunks` có ít nhất 500 vectors sau test run
- [ ] Qdrant collection `image_patches` có ít nhất 500 vectors sau test run
- [ ] Query test: `client.search("text_chunks", query_vector, limit=5)` trả về kết quả có nghĩa
- [ ] Query test: `client.search("image_patches", image_vector, limit=5)` trả về ảnh tương tự
- [ ] Chạy full ingestion trên toàn bộ data → ghi thời gian và số lượng records vào notebook
- [ ] Commit: "feat: ingestion and indexing pipeline"

---

### ✅ MILESTONE 4 — Baseline RAG (Config A)
**Deadline: Tuần 3, Ngày 1–4**

**Build checklist:**
- [ ] Implement vanilla RAG trong `notebooks/02_baseline_rag.ipynb`:
  - Nhận query → BioMedBERT encode → Qdrant search → lấy top-3 text chunks
  - Nếu có ảnh → BioCLIP encode → Qdrant search → lấy top-3 image patches
  - Concat context → Qwen2.5-VL generate answer (không có agent, không có rerank)
- [ ] Chạy trên **200 VQA-RAD test samples**
- [ ] Chạy trên **100 MedQA test samples**
- [ ] Tính metrics: Accuracy (exact match), Hit Rate@5, Faithfulness (RAGAS)

**Review checklist khi hoàn thành:**
- [ ] File `experiments/results/config_A_baseline.json` tồn tại với đầy đủ per-sample results
- [ ] Summary metrics được ghi vào `README.md` phần "Results"
- [ ] Baseline accuracy trên VQA-RAD được ghi nhận (đây là con số cần vượt qua)
- [ ] Không có data leakage: test samples không có trong index
- [ ] Commit: "feat: baseline RAG (Config A) evaluation"

---

### ✅ MILESTONE 5 — LangGraph Agent Core
**Deadline: Tuần 3, Ngày 5 — Tuần 4, Ngày 3**

**Build checklist:**
- [ ] Implement `src/agents/state.py` — `AgentState` TypedDict đầy đủ fields
- [ ] Implement `src/agents/decomposer.py` — `decompose_query()` node
  - Prompt template cho decomposition
  - Handle 3 cases: text-only, visual-only, hybrid
- [ ] Implement `src/agents/text_retriever.py`:
  - Dense retrieval từ Qdrant
  - BM25 scoring
  - Hybrid fusion (0.7 dense + 0.3 BM25)
  - BGE-reranker cross-encoding → top-5
- [ ] Implement `src/agents/visual_retriever.py`:
  - Stage 1: BioCLIP coarse search → top-10
  - Stage 2: Qwen2.5-VL `ground_region()` → crop ROI
  - Caption generation cho regions
  - Score lại → top-3
- [ ] Implement `src/agents/decision.py`:
  - `fuse_and_verify()`
  - `generate_answer()` với citations
- [ ] Implement `src/agents/graph.py` — StateGraph với conditional routing

**Review checklist khi hoàn thành:**
- [ ] `graph.invoke({"question": "...", "image_path": "..."})` chạy end-to-end không lỗi
- [ ] State sau mỗi node được log — kiểm tra `step_count` tăng đúng
- [ ] Kiểm tra routing: text-only query không gọi `retrieve_visual`
- [ ] Kiểm tra routing: hybrid query gọi cả 2 retrieval nodes
- [ ] `state["citations"]` không rỗng trong kết quả cuối
- [ ] Chạy 10 VQA-RAD samples qua agent — kết quả hợp lý
- [ ] Commit: "feat: LangGraph agent system"

---

### ✅ MILESTONE 6 — Fine-grained Visual Retrieval
**Deadline: Tuần 4, Ngày 4–7**

**Build checklist:**
- [ ] Nâng cấp `src/agents/visual_retriever.py`:
  - Implement `ground_region()` trong `src/models/qwen_vl.py` đầy đủ
  - Crop ảnh theo bbox từ grounding
  - Re-encode cropped region qua BioCLIP
  - Score theo caption similarity với visual_subquery
- [ ] Tạo **Config C**: fine-grained retrieval nhưng không có agent (single-step)
  - Chạy 200 VQA-RAD samples
  - Ghi results vào `config_C_finegrained_only.json`
- [ ] Tạo **Config B**: agent nhưng standard retrieval (disable fine-grained stage 2)
  - Chạy 200 VQA-RAD samples
  - Ghi results vào `config_B_agent_only.json`

**Review checklist khi hoàn thành:**
- [ ] `ground_region()` trả về bbox hợp lệ (0 ≤ x1 < x2 ≤ 1, 0 ≤ y1 < y2 ≤ 1)
- [ ] Cropped regions được lưu tạm vào `data/mimic_cxr/processed/crops/` để debug
- [ ] Visual retrieval với fine-grained (Stage 2) cho Hit Rate cao hơn coarse-only — verify trên 50 samples
- [ ] `config_B_agent_only.json` và `config_C_finegrained_only.json` tồn tại
- [ ] Commit: "feat: fine-grained visual retrieval and ablation configs B/C"

---

### ✅ MILESTONE 7 — Full Evaluation & Ablation Study
**Deadline: Tuần 5–6**

**Build checklist:**
- [ ] Implement `src/evaluation/benchmark.py` — full eval loop
- [ ] Implement tất cả metrics trong `src/evaluation/metrics.py`
- [ ] Chạy **Config D (Full System)** trên toàn bộ eval set:
  - 200 VQA-RAD
  - 100 MedQA
  - 50 MIMIC-CXR
- [ ] Tổng hợp kết quả 4 configs vào `notebooks/04_ablation_study.ipynb`
- [ ] Vẽ bảng so sánh 4 configs × 3 datasets × 4 metrics
- [ ] Thực hiện error analysis: phân loại 20–30 failure cases

**Review checklist khi hoàn thành:**
- [ ] Config D accuracy VQA-RAD > Config A + 10% (target)
- [ ] Config D Hit Rate@5 > Config A (target)
- [ ] Tất cả 4 config JSON results tồn tại và đầy đủ
- [ ] Error analysis phân loại được ít nhất 3 loại lỗi chính:
  - Retrieval failure (đúng answer nhưng không retrieve được)
  - Reasoning failure (retrieve đúng nhưng generate sai)
  - Grounding failure (visual retrieval không chính xác)
- [ ] Commit: "feat: full evaluation and ablation study"

---

### ✅ MILESTONE 8 — Gradio Demo
**Deadline: Tuần 7**

**Build checklist:**
- [ ] Implement `demo/app.py` với Gradio:
  - Input: text question + optional image upload
  - Output: answer + retrieved evidence + reasoning steps + citations
  - Hiển thị cropped ROI nếu có visual retrieval
- [ ] Test locally
- [ ] Deploy lên HuggingFace Spaces (free tier)
- [ ] Ghi URL demo vào `README.md`

**Review checklist khi hoàn thành:**
- [ ] Demo chạy được trên máy local
- [ ] Upload 1 ảnh chest X-ray + câu hỏi → nhận được answer có grounding
- [ ] HuggingFace Spaces URL hoạt động (public)
- [ ] Demo URL được ghi vào `README.md` và báo cáo
- [ ] Commit: "feat: Gradio demo deployment"

---

### ✅ MILESTONE 9 — Báo cáo Đồ án 2
**Deadline: Tuần 8–12 (song song với code)**

**Cấu trúc báo cáo:**
- [ ] **Chương 1 — Giới thiệu** (5–8 trang)
  - [ ] Problem statement: tại sao RAG thông thường không đủ cho medical VQA
  - [ ] Motivation: fine-grained retrieval + agentic reasoning
  - [ ] Mục tiêu cụ thể và measurable (accuracy target, dataset)
  - [ ] Phạm vi: inference-only, không RL, domain radiology+pathology
  - [ ] Đóng góp chính (3 điểm như đã soạn)

- [ ] **Chương 2 — Tổng quan nghiên cứu** (15–20 trang)
  - [ ] Taxonomy 3 chiều: Retrieval granularity × Agent architecture × Training paradigm
  - [ ] Bảng so sánh 15–20 paper (Paper | Venue | Granularity | Agent | Medical | Dataset | Key Metric)
  - [ ] Papers phải đọc: HM-RAG, VimRAG, A-MAR, MMed-RAG, VRAG-RL, Survey Multimodal RAG
  - [ ] Phân tích gap: chưa có paper nào hybrid B+A trên medical domain

- [ ] **Chương 3 — Thiết kế hệ thống** (10–15 trang)
  - [ ] Architecture diagram (4 layers)
  - [ ] LangGraph state machine diagram
  - [ ] Fine-grained visual retrieval flow (2 stages)
  - [ ] Data pipeline diagram

- [ ] **Chương 4 — Triển khai & Thực nghiệm** (15–20 trang)
  - [ ] Dataset statistics table
  - [ ] Implementation details (tech stack, hyperparameters)
  - [ ] Ablation study table (4 configs × metrics)
  - [ ] Error analysis + case studies (5 đúng + 5 sai)
  - [ ] Inference time benchmark

- [ ] **Chương 5 — Kết luận & Hướng phát triển** (5–8 trang)
  - [ ] Tóm tắt kết quả
  - [ ] Hạn chế hiện tại
  - [ ] Hướng cho Đồ án TN: GraphRAG + RL fine-tuning + tiếng Việt

- [ ] **Phụ lục**
  - [ ] GitHub URL
  - [ ] HuggingFace Spaces demo URL
  - [ ] Dataset access information (PhysioNet credentialed note cho MIMIC-CXR)

---

## 7. LƯU Ý QUAN TRỌNG CHO AI CODING AGENT

### Không được làm
1. **Không thay đổi vector dimensions** sau khi đã index — sẽ phải xóa collection và index lại
2. **Không dùng Chroma** — dùng Qdrant (multi-vector support tốt hơn)
3. **Không để test data trong index** — data leakage làm kết quả vô nghĩa
4. **Không quên pin LangGraph version** — breaking changes giữa versions
5. **Không load tất cả data vào RAM** — dùng batch processing (batch_size=32)

### Phải làm
1. **Ghi log mọi kết quả** kể cả thất bại — `experiments/results/` là source of truth
2. **Commit thường xuyên** sau mỗi milestone — ít nhất 1 commit/ngày khi đang code
3. **Kiểm tra MIMIC-CXR license** — ghi rõ trong báo cáo "used under PhysioNet credentialed access for academic research"
4. **Document hyperparameters** — mỗi lần chạy eval ghi đầy đủ config vào JSON output
5. **Giữ reproducibility** — set random seed, ghi Python + library versions vào `requirements.txt`

### Thứ tự ưu tiên nếu thiếu thời gian
1. Milestone PRE-0 (bắt buộc): Đọc papers + duyệt repos
2. Milestone 0–4 (bắt buộc): Setup + Data + Baseline
3. Milestone 5 (bắt buộc): Agent core
4. Milestone 7 (bắt buộc): Evaluation
5. Milestone 6 (nên có): Fine-grained visual
6. Milestone 8 (tốt có): Demo
7. Milestone 9 (bắt buộc): Báo cáo

---

## 8. TIMELINE TỔNG QUAN

| Tháng | Tuần | Milestone | Output |
|---|---|---|---|
| Tháng 1 | W1 D1 | PRE-0 | Papers đọc xong, notes ghi lại |
| Tháng 1 | W1 | M0, M1 | Env setup, normalized data |
| Tháng 1 | W2 | M2, M3 | Models, Qdrant index |
| Tháng 1–2 | W3–4 | M4, M5 | Baseline + Agent core |
| Tháng 2 | W5–6 | M6 | Fine-grained visual |
| Tháng 2–3 | W7–8 | M7 | Full eval + ablation |
| Tháng 3 | W9 | M8 | Demo |
| Tháng 3–5 | W10–20 | M9 | Báo cáo (song song) |

---

## 9. QUICK REFERENCE — COMMANDS

```bash
# Start Qdrant
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Run full ingestion
python -m src.ingestion.indexer --datasets vqa_rad roco mimic_cxr medqa bioasq pathvqa

# Run baseline eval (Config A)
python -m src.evaluation.benchmark --config A --datasets vqa_rad medqa --output experiments/results/

# Run full agent eval (Config D)
python -m src.evaluation.benchmark --config D --datasets vqa_rad medqa mimic_cxr --output experiments/results/

# Launch demo
cd demo && gradio app.py

# Run tests
pytest tests/ -v
```

---

*File này được viết cho AI Coding Agent. Mỗi khi hoàn thành một Milestone, check off tất cả checkboxes trong milestone đó trước khi chuyển sang milestone tiếp theo.*