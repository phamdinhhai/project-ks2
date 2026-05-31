"""Fine-grained chunking for text and images.

Text: sentence-level sliding window chunks.
Image: 5 patches (full + 4 quadrants) per image — inspired by VimRAG.

Reference: VimRAG (src/support_repo/VimRAG_project) uses patch-level image
representations to enable region-specific visual retrieval.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# Simple sentence splitter that handles common medical abbreviations
_SENT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|"  # standard sentence boundary
    r"(?<=\n)\s*(?=\S)",  # newline boundary
    re.UNICODE,
)
# Abbreviations that should NOT trigger a split
_ABBREV = {"dr.", "mr.", "mrs.", "ms.", "vs.", "etc.", "e.g.", "i.e.", "fig.", "no."}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting common medical abbreviations."""
    parts = _SENT_RE.split(text)
    sentences = []
    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        combined = f"{buffer} {part}".strip() if buffer else part
        # Check if we are mid-abbreviation
        lower = combined.lower()
        if any(lower.endswith(abbr) for abbr in _ABBREV):
            buffer = combined
        else:
            sentences.append(combined)
            buffer = ""
    if buffer:
        sentences.append(buffer)
    return sentences


def chunk_text(
    text: str,
    window_size: int = 3,
    overlap: int = 1,
) -> list[str]:
    """Chunk text into sentence-level sliding window segments.

    Args:
        text: input document text
        window_size: number of sentences per chunk
        overlap: number of overlapping sentences between consecutive chunks

    Returns:
        list of text chunks
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    step = max(1, window_size - overlap)
    chunks = []
    for start in range(0, len(sentences), step):
        window = sentences[start : start + window_size]
        chunks.append(" ".join(window))
        if start + window_size >= len(sentences):
            break
    return chunks


def chunk_image(image_path: str | Path) -> list[dict[str, Any]]:
    """Split a medical image into 5 patches: full + 4 quadrants.

    This enables fine-grained visual retrieval at region level rather than
    whole-image level, critical for radiology (e.g. localizing pneumonia opacity).

    Args:
        image_path: path to image file

    Returns:
        list of dicts with keys: patch_id, image (PIL.Image), bbox (x1,y1,x2,y2 relative 0-1)
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    patches = [
        {"patch_id": "full", "bbox": (0.0, 0.0, 1.0, 1.0), "image": img},
    ]

    # 4 quadrants
    mid_w = width // 2
    mid_h = height // 2
    quadrants = [
        ("tl", (0, 0, mid_w, mid_h), (0.0, 0.0, 0.5, 0.5)),
        ("tr", (mid_w, 0, width, mid_h), (0.5, 0.0, 1.0, 0.5)),
        ("bl", (0, mid_h, mid_w, height), (0.0, 0.5, 0.5, 1.0)),
        ("br", (mid_w, mid_h, width, height), (0.5, 0.5, 1.0, 1.0)),
    ]
    for patch_id, pixel_box, rel_bbox in quadrants:
        patch = img.crop(pixel_box)
        patches.append({
            "patch_id": patch_id,
            "bbox": rel_bbox,
            "image": patch,
        })

    return patches
