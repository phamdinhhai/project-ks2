from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from medical_rag.data_loaders import load_corpora
from medical_rag.schema import DocumentChunk, ImageRecord
from medical_rag.text_utils import tokenize

INDEX_FILE = "rag_indexes.joblib"


class RAGIndexes(dict):
    """Dictionary-backed index bundle with named keys for simple persistence."""


def _dataset_stats(documents: list[DocumentChunk], images: list[ImageRecord]) -> dict[str, dict[str, int]]:
    doc_counts = Counter(doc.dataset for doc in documents)
    image_counts = Counter(img.dataset for img in images)
    datasets = sorted(set(doc_counts) | set(image_counts))
    return {
        dataset: {
            "documents": int(doc_counts.get(dataset, 0)),
            "images": int(image_counts.get(dataset, 0)),
        }
        for dataset in datasets
    }


def _build_aliases(documents: list[DocumentChunk], images: list[ImageRecord]) -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)

    for idx, doc in enumerate(documents):
        record_id = str(doc.metadata.get("record_id") or "")
        base_record_id = str(doc.metadata.get("base_record_id") or "")
        values = {
            doc.id,
            f"text-{idx}",
            record_id,
            base_record_id,
            f"{doc.dataset}-{record_id}" if record_id else "",
            f"{doc.dataset}-text-{record_id}" if record_id else "",
        }
        for value in values:
            if value:
                aliases[value].add(doc.id)

    for idx, image in enumerate(images):
        record_id = str(image.metadata.get("record_id") or "")
        base_record_id = str(image.metadata.get("base_record_id") or "")
        values = {
            image.id,
            f"image-{idx}",
            record_id,
            base_record_id,
            f"{image.dataset}-{record_id}" if record_id else "",
            f"{image.dataset}-image-{record_id}" if record_id else "",
        }
        for value in values:
            if value:
                aliases[value].add(image.id)

    return {key: sorted(value) for key, value in aliases.items()}


def build_indexes(data_dir: Path, index_dir: Path, limit_per_dataset: int | None = None) -> dict[str, Any]:
    documents, images = load_corpora(data_dir, limit_per_dataset=limit_per_dataset)
    index_dir.mkdir(parents=True, exist_ok=True)

    text_corpus = [doc.text for doc in documents]
    text_tokens = [tokenize(text) for text in text_corpus]
    text_bm25 = BM25Okapi(text_tokens) if text_tokens else None
    text_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=50000)
    text_matrix = text_vectorizer.fit_transform(text_corpus) if text_corpus else None

    image_corpus = [img.caption for img in images]
    image_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=50000)
    image_matrix = image_vectorizer.fit_transform(image_corpus) if image_corpus else None

    dataset_stats = _dataset_stats(documents, images)
    id_aliases = _build_aliases(documents, images)

    bundle = {
        "documents": documents,
        "images": images,
        "text_bm25": text_bm25,
        "text_vectorizer": text_vectorizer if text_corpus else None,
        "text_matrix": text_matrix,
        "image_vectorizer": image_vectorizer if image_corpus else None,
        "image_matrix": image_matrix,
        "dataset_stats": dataset_stats,
        "id_aliases": id_aliases,
    }
    joblib.dump(bundle, index_dir / INDEX_FILE)
    return bundle


def load_indexes(index_dir: Path) -> dict[str, Any]:
    index_path = index_dir / INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}. Run build-index first.")
    return joblib.load(index_path)


def index_stats(bundle: dict[str, Any]) -> dict[str, Any]:
    documents: list[DocumentChunk] = bundle.get("documents", [])
    images: list[ImageRecord] = bundle.get("images", [])
    return {
        "documents": len(documents),
        "images": len(images),
        "datasets": bundle.get("dataset_stats", {}),
        "id_aliases": len(bundle.get("id_aliases", {})),
    }
