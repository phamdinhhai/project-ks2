from __future__ import annotations

import re

from medical_rag.schema import Language, Modality, QueryIntent


VI_MARKERS = {
    "đ", "ă", "â", "ê", "ô", "ơ", "ư",
    "bệnh", "triệu", "điều", "trị", "hình", "ảnh", "phổi", "là", "của",
}
IMAGE_MARKERS = {
    "image", "images", "xray", "x-ray", "ct", "mri", "scan", "figure", "visual",
    "ảnh", "hình", "x quang", "x-quang", "chụp", "phim", "ct", "mri",
}
DATASET_MARKERS = {
    "medqa": "medqa",
    "bioasq": "bioasq",
    "vqa-rad": "vqa_rad",
    "vqa_rad": "vqa_rad",
    "roco": "roco",
    "mimic": "mimic_cxr",
    "cxr": "mimic_cxr",
}
IMAGE_DATASETS = {"vqa_rad", "roco", "mimic_cxr"}
TEXT_ONLY_DATASETS = {"medqa", "bioasq"}


class QueryRouter:
    """Lightweight deterministic query router inspired by HM-RAG/VimRAG routing."""

    def detect_language(self, query: str) -> Language:
        lowered = query.lower()
        if any(marker in lowered for marker in VI_MARKERS):
            return Language.VI
        if re.search(r"[a-zA-Z]", query):
            return Language.EN
        return Language.UNKNOWN

    def detect_dataset_hint(self, query: str) -> str | None:
        lowered = query.lower()
        for marker, dataset in DATASET_MARKERS.items():
            if marker in lowered:
                return dataset
        return None

    def detect_modality(self, query: str, image_path: str | None = None) -> Modality:
        if image_path:
            return Modality.MIXED if query.strip() else Modality.IMAGE
        lowered = query.lower()
        if any(marker in lowered for marker in IMAGE_MARKERS):
            return Modality.IMAGE
        return Modality.TEXT

    def route(self, query: str, image_path: str | None = None, dataset_hint: str | None = None) -> QueryIntent:
        language = self.detect_language(query)
        inferred_dataset = dataset_hint or self.detect_dataset_hint(query)
        modality = self.detect_modality(query, image_path=image_path)
        reasons = [f"language={language.value}", f"modality={modality.value}"]

        use_image = modality in {Modality.IMAGE, Modality.MIXED}
        if inferred_dataset in IMAGE_DATASETS:
            use_image = True
            reasons.append(f"dataset {inferred_dataset} supports image branch")
        if inferred_dataset in TEXT_ONLY_DATASETS:
            use_image = False
            reasons.append(f"dataset {inferred_dataset} is text-only")

        return QueryIntent(
            query=query,
            language=language,
            modality=modality,
            dataset_hint=inferred_dataset,
            use_text_branch=True,
            use_image_branch=use_image,
            reasons=reasons,
        )
