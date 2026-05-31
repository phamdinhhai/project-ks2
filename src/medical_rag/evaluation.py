from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from medical_rag.pipeline import MedicalRAGPipeline
from medical_rag.schema import FusedEvidence, Modality


def _aliases_for_evidence(item: FusedEvidence, index: int) -> set[str]:
    metadata = item.metadata or {}
    record_id = str(metadata.get("record_id") or "")
    base_record_id = str(metadata.get("base_record_id") or "")
    aliases = {
        item.id,
        record_id,
        base_record_id,
        f"{item.dataset}-{record_id}" if record_id else "",
        f"{item.dataset}-{item.modality.value}-{record_id}" if record_id else "",
        f"{item.modality.value}-{index}",
    }
    return {alias for alias in aliases if alias}


def _match_gold(
    gold_id: str | None,
    evidence: list[FusedEvidence],
    alias_map: dict[str, list[str]],
) -> tuple[bool, float, str | None, bool]:
    if not gold_id:
        return False, 0.0, None, False

    canonical_ids = set(alias_map.get(gold_id, []))
    for rank, item in enumerate(evidence, start=1):
        aliases = _aliases_for_evidence(item, rank - 1)
        if gold_id in aliases:
            return True, 1.0 / rank, item.id, gold_id != item.id
        if item.id in canonical_ids:
            return True, 1.0 / rank, item.id, True

    for rank, item in enumerate(evidence, start=1):
        if item.id.endswith(gold_id) or gold_id.endswith(item.id):
            return True, 1.0 / rank, item.id, True

    return False, 0.0, None, False


def _safe_div(numerator: float, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _answer_accuracy(pred: str, gold: str | None) -> float:
    if not gold:
        return 0.0
    p = str(pred).strip().lower()
    g = str(gold).strip().lower()
    if p == g or g in p or p in g:
        return 1.0
    return 0.0


def _error_categories(
    expected_modality: str,
    route_ok: bool,
    hit: bool,
    evidence: list[FusedEvidence],
    alias_only: bool,
    ans_acc: float,
) -> list[str]:
    errors: list[str] = []
    if not route_ok:
        errors.append("routing_failure")
    if not hit:
        errors.append("retrieval_failure")
    if expected_modality == "image" and not any(item.modality == Modality.IMAGE for item in evidence):
        errors.append("grounding_failure")
    if hit and ans_acc < 1.0:
        errors.append("reasoning_failure")
    if alias_only:
        errors.append("alias_only_match")
    return errors or ["ok"]


def evaluate(eval_file: Path, pipeline: MedicalRAGPipeline, top_k: int = 5) -> dict:
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    rows = []

    alias_map = pipeline.index_bundle.get("id_aliases", {})

    recall_hits = 0
    mrr_total = 0.0
    route_hits = 0
    image_branch_used = 0
    fallback_count = 0
    evidence_count_total = 0
    alias_only_matches = 0
    answer_acc_total = 0.0
    answer_cases = 0

    text_cases = 0
    text_hits = 0
    text_mrr = 0.0

    image_cases = 0
    image_hits = 0
    image_mrr = 0.0

    modality_distribution: Counter[str] = Counter()
    dataset_distribution: Counter[str] = Counter()
    error_distribution: Counter[str] = Counter()

    for case in cases:
        answer = pipeline.run(
            case["query"],
            top_k=top_k,
            dataset_hint=case.get("dataset_hint"),
        )
        evidence = answer.evidence[:top_k]
        evidence_ids = [item.id for item in evidence]
        evidence_count_total += len(evidence)
        for item in evidence:
            modality_distribution[item.modality.value] += 1
            dataset_distribution[item.dataset] += 1

        gold = case.get("gold_text_id") or case.get("gold_image_id")
        hit, rr, matched_id, alias_only = _match_gold(gold, evidence, alias_map)
        recall_hits += int(hit)
        mrr_total += rr
        alias_only_matches += int(alias_only)

        gold_answer = case.get("gold_answer")
        ans_acc = 0.0
        if gold_answer:
            ans_acc = _answer_accuracy(answer.answer, gold_answer)
            answer_acc_total += ans_acc
            answer_cases += 1

        expected_modality = str(case.get("modality") or "")
        route_ok = expected_modality == answer.intent.modality.value or (
            expected_modality == "image" and answer.intent.use_image_branch
        )
        route_hits += int(route_ok)
        image_branch_used += int(answer.intent.use_image_branch)
        fallback_count += int(any("fallback" in reason.lower() for reason in answer.intent.reasons))

        if expected_modality == "text":
            text_cases += 1
            text_hits += int(hit)
            text_mrr += rr
        elif expected_modality == "image":
            image_cases += 1
            image_hits += int(hit)
            image_mrr += rr

        error_categories = _error_categories(expected_modality, route_ok, hit, evidence, alias_only, ans_acc)
        error_distribution.update(error_categories)

        rows.append({
            "query": case["query"],
            "gold": gold,
            "gold_answer": gold_answer,
            "pred_answer": answer.answer,
            "answer_accuracy": ans_acc,
            "matched_id": matched_id,
            "evidence_ids": evidence_ids,
            "hit": hit,
            "mrr": rr,
            "alias_only_match": alias_only,
            "route": answer.intent.modality.value,
            "use_image_branch": answer.intent.use_image_branch,
            "evidence_modalities": [item.modality.value for item in evidence],
            "evidence_datasets": [item.dataset for item in evidence],
            "error_categories": error_categories,
        })

    total = len(cases)
    return {
        "total": total,
        "recall_at_k": _safe_div(recall_hits, total),
        "hit_rate_at_k": _safe_div(recall_hits, total),
        "mrr_at_k": _safe_div(mrr_total, total),
        "answer_accuracy": _safe_div(answer_acc_total, answer_cases),
        "routing_accuracy": _safe_div(route_hits, total),
        "image_branch_used_rate": _safe_div(image_branch_used, total),
        "fallback_rate": _safe_div(fallback_count, total),
        "mean_evidence_count": _safe_div(evidence_count_total, total),
        "alias_only_match_rate": _safe_div(alias_only_matches, total),
        "evidence_modality_distribution": dict(modality_distribution),
        "dataset_distribution": dict(dataset_distribution),
        "error_distribution": dict(error_distribution),
        "per_modality": {
            "text": {
                "total": text_cases,
                "recall_at_k": _safe_div(text_hits, text_cases),
                "mrr_at_k": _safe_div(text_mrr, text_cases),
            },
            "image": {
                "total": image_cases,
                "recall_at_k": _safe_div(image_hits, image_cases),
                "mrr_at_k": _safe_div(image_mrr, image_cases),
            },
        },
        "rows": rows,
    }
