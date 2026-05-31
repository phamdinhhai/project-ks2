# Data Model Reference — Pydantic Schemas

All schemas defined in `src/medical_rag/schema.py`.

## Enums

### `Language`
```python
class Language(str, Enum):
    EN = "en"   # English
    VI = "vi"   # Vietnamese
```

### `Modality`
```python
class Modality(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
```

## Core Document Types

### `DocumentChunk` — Text record for indexing/retrieval
```python
class DocumentChunk(BaseModel):
    id: str                # "{dataset}-text-{record_id}"
    text: str              # Full document/chunk text
    dataset: str           # "medqa" | "bioasq" | "vqa_rad" | ...
    source_path: str       # Path to original JSONL
    modality: Modality = Modality.TEXT
    metadata: dict         # record_id, question, answer, ...
```

### `ImageRecord` — Image record for indexing/retrieval
```python
class ImageRecord(BaseModel):
    id: str                # "{dataset}-image-{record_id}"
    caption: str           # Image caption or report text
    dataset: str
    image_path: str | None # Relative path to image file
    modality: Modality = Modality.IMAGE
    metadata: dict
```

### `RetrievalResult` — Output from a retriever
```python
class RetrievalResult(BaseModel):
    id: str
    text: str
    dataset: str
    modality: Modality
    score: float           # Raw retriever score
    rank: int              # 1-indexed rank within results
    source_path: str
    metadata: dict
```

### `FusedEvidence` — After late fusion
```python
class FusedEvidence(BaseModel):
    id: str
    modality: Modality
    fused_score: float     # RRF or weighted combined score
    text: str
    dataset: str
    source_path: str
    component_scores: dict # {"text_rrf": 0.01, "image_rrf": 0.005, ...}
    metadata: dict
```

### `QueryIntent` — Router output
```python
class QueryIntent(BaseModel):
    query: str
    language: Language
    use_image_branch: bool  # True → activate image retrieval
    dataset_hint: str | None
    reasons: list[str]      # Routing decision trace
    modality: str           # "text" | "image" | "text+image"
```

### `GeneratedAnswer` — Final pipeline output
```python
class GeneratedAnswer(BaseModel):
    answer: str            # Answer text (extractive or LLM-generated)
    intent: QueryIntent
    evidence: list[FusedEvidence]
    citations: list[str]   # ["[1] dataset:id (modality)", ...]
```

## Agent State (LangGraph)

Defined in `src/medical_rag/agents/state.py`:

```python
class AgentState(TypedDict, total=False):
    question: str
    image_path: str | None
    text_subquery: str
    visual_subquery: str
    query_type: str            # "text" | "visual" | "hybrid"
    text_evidence: list[dict]  # [{id, text, score, dataset, ...}]
    visual_evidence: list[dict]# [{id, image_path, patch_id, bbox, score, ...}]
    fused_evidence: str        # Formatted context for VLM
    faithfulness_score: float
    answer: str
    reasoning_steps: list[str]
    citations: list[str]
    step_count: int
    error: str | None
    dataset_hint: str | None
    config: dict
```

## Canonical JSONL Row (on-disk format)

```json
{
  "dataset": "vqa_rad",
  "split": "train",
  "record_id": "img_001",
  "question": "What is shown in the image?",
  "answer": "Cardiomegaly",
  "text": "What is shown in the image?\nCardiomegaly",
  "image_path": "vqa_rad/images/train/img_001.jpg",
  "metadata": {"answer_type": "CLOSED", "body_part": "CHEST"}
}
```
