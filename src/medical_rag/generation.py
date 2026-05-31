from __future__ import annotations

from medical_rag.schema import FusedEvidence, GeneratedAnswer, Modality, QueryIntent


class ExtractiveGenerator:
    """Offline answer generator that summarizes retrieved evidence with citations."""

    def generate(self, intent: QueryIntent, evidence: list[FusedEvidence]) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer(
                answer="Không tìm thấy bằng chứng phù hợp trong corpus hiện có.",
                intent=intent,
                evidence=[],
                citations=[],
            )

        citations = [f"[{idx}] {item.dataset}:{item.id} ({item.modality.value})" for idx, item in enumerate(evidence, start=1)]
        snippets = []
        for idx, item in enumerate(evidence[:4], start=1):
            clean = " ".join(item.text.split())
            source_label = "image/caption" if item.modality == Modality.IMAGE else "text"
            snippets.append(f"[{idx}] ({source_label}) {clean[:420]}")

        has_image_evidence = any(item.modality == Modality.IMAGE for item in evidence)
        if not intent.use_image_branch:
            branch_note = "text-only"
        elif has_image_evidence:
            branch_note = "text + image/caption"
        else:
            branch_note = "text fallback; image evidence weak or unavailable"

        reasons = "\n".join(f"- {reason}" for reason in intent.reasons)
        answer = (
            f"Query route: {intent.language.value}, {branch_note}.\n"
            f"Routing reasons:\n{reasons}\n\n"
            "Evidence-based answer draft:\n"
            + "\n".join(snippets)
            + "\n\nCitations: "
            + "; ".join(citations)
        )
        return GeneratedAnswer(answer=answer, intent=intent, evidence=evidence, citations=citations)
