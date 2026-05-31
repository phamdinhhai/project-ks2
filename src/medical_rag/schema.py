from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Language(str, Enum):
    EN = "en"
    VI = "vi"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"


class QueryIntent(BaseModel):
    query: str
    language: Language
    modality: Modality
    dataset_hint: str | None = None
    use_text_branch: bool = True
    use_image_branch: bool = False
    reasons: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    id: str
    text: str
    dataset: str
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageRecord(BaseModel):
    id: str
    caption: str
    dataset: str
    image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    id: str
    modality: Modality
    score: float
    rank: int
    text: str
    dataset: str
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusedEvidence(BaseModel):
    id: str
    modality: Modality
    fused_score: float
    text: str
    dataset: str
    source_path: str | None = None
    component_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    answer: str
    intent: QueryIntent
    evidence: list[FusedEvidence]
    citations: list[str]
