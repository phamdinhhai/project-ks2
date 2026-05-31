"""Advanced evaluation metrics for the Medical RAG pipeline.

Adds on top of the baseline evaluation.py:
- Exact Match + Token-level F1 answer accuracy
- Per-dataset metrics breakdown
- NDCG@k for retrieval
- Agent pipeline evaluation (wraps AgenticRAGPipeline)
- RAGAS integration (optional, requires ragas package)

Usage:
  from medical_rag.evaluation_advanced import evaluate_advanced, evaluate_agent
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Token-level metrics
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, gold: str) -> float:
    """Binary exact match after normalization."""
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 score between predicted and gold answers."""
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _safe_div(a: float, b: int | float) -> float:
    return float(a) / float(b) if b else 0.0


# ---------------------------------------------------------------------------
#  NDCG@k
# ---------------------------------------------------------------------------

def ndcg_at_k(relevant_positions: list[int], k: int) -> float:
    """Compute NDCG@k given 1-indexed positions of relevant docs.

    Args:
        relevant_positions: list of 1-indexed positions where relevant docs were found
        k: cutoff

    Returns:
        NDCG@k score
    """
    import math

    dcg = 0.0
    for pos in relevant_positions:
        if pos <= k:
            dcg += 1.0 / math.log2(pos + 1)

    # Ideal DCG: all relevant at top
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_positions), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
#  Per-dataset breakdown
# ---------------------------------------------------------------------------

def _compute_per_dataset(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Group evaluation rows by dataset and compute metrics per group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        ds = row.get("dataset_hint") or row.get("dataset") or "unknown"
        groups[ds].append(row)

    per_ds: dict[str, dict[str, float]] = {}
    for ds, ds_rows in sorted(groups.items()):
        total = len(ds_rows)
        hits = sum(1 for r in ds_rows if r.get("hit"))
        mrr = sum(r.get("mrr", 0.0) for r in ds_rows)
        em = sum(r.get("exact_match", 0.0) for r in ds_rows)
        f1 = sum(r.get("token_f1", 0.0) for r in ds_rows)
        ans_cases = sum(1 for r in ds_rows if r.get("gold_answer"))

        per_ds[ds] = {
            "total": total,
            "recall_at_k": _safe_div(hits, total),
            "mrr_at_k": _safe_div(mrr, total),
            "exact_match": _safe_div(em, ans_cases),
            "token_f1": _safe_div(f1, ans_cases),
        }
    return per_ds


# ---------------------------------------------------------------------------
#  Main advanced evaluate — baseline pipeline
# ---------------------------------------------------------------------------

def evaluate_advanced(
    eval_file: Path,
    pipeline: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run advanced evaluation on the baseline MedicalRAGPipeline.

    Adds exact_match, token_f1, per-dataset breakdown, NDCG on top
    of the baseline metrics from evaluation.py.

    Args:
        eval_file: path to eval_cases.json
        pipeline: MedicalRAGPipeline instance
        top_k: number of evidence items

    Returns:
        dict with all metrics + per_dataset + rows
    """
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    alias_map = pipeline.index_bundle.get("id_aliases", {})

    rows: list[dict] = []
    totals: dict[str, float] = defaultdict(float)
    counters: dict[str, int] = defaultdict(int)
    error_dist: Counter[str] = Counter()

    for case in cases:
        answer = pipeline.run(
            case["query"],
            top_k=top_k,
            dataset_hint=case.get("dataset_hint"),
        )
        evidence = answer.evidence[:top_k]

        gold = case.get("gold_text_id") or case.get("gold_image_id")
        gold_answer = case.get("gold_answer")

        # Hit / MRR
        from medical_rag.evaluation import _aliases_for_evidence, _match_gold
        hit, rr, matched_id, alias_only = _match_gold(gold, evidence, alias_map)

        # Answer metrics
        em = exact_match(answer.answer, gold_answer) if gold_answer else 0.0
        f1 = token_f1(answer.answer, gold_answer) if gold_answer else 0.0

        # Routing
        expected_mod = str(case.get("modality") or "")
        route_ok = expected_mod == answer.intent.modality.value or (
            expected_mod == "image" and answer.intent.use_image_branch
        )

        # Error categories
        from medical_rag.evaluation import _error_categories
        errors = _error_categories(expected_mod, route_ok, hit, evidence, alias_only, em)
        error_dist.update(errors)

        row = {
            "query": case["query"],
            "dataset_hint": case.get("dataset_hint") or case.get("dataset"),
            "dataset": case.get("dataset"),
            "gold": gold,
            "gold_answer": gold_answer,
            "pred_answer": answer.answer,
            "hit": hit,
            "mrr": rr,
            "exact_match": em,
            "token_f1": f1,
            "route_ok": route_ok,
            "alias_only": alias_only,
            "matched_id": matched_id,
            "error_categories": errors,
            "modality": expected_mod,
        }
        rows.append(row)

        totals["recall"] += int(hit)
        totals["mrr"] += rr
        totals["route"] += int(route_ok)
        totals["em"] += em
        totals["f1"] += f1
        counters["total"] += 1
        if gold_answer:
            counters["ans_cases"] += 1

    total = counters["total"]
    ans_cases = counters["ans_cases"]

    metrics = {
        "total": total,
        "recall_at_k": _safe_div(totals["recall"], total),
        "mrr_at_k": _safe_div(totals["mrr"], total),
        "routing_accuracy": _safe_div(totals["route"], total),
        "exact_match": _safe_div(totals["em"], ans_cases),
        "token_f1": _safe_div(totals["f1"], ans_cases),
        "error_distribution": dict(error_dist),
        "per_dataset": _compute_per_dataset(rows),
        "rows": rows,
    }
    return metrics


# ---------------------------------------------------------------------------
#  Agent pipeline evaluate
# ---------------------------------------------------------------------------

def evaluate_agent(
    eval_file: Path,
    agent_pipeline: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run evaluation on the AgenticRAGPipeline.

    The agent pipeline returns dicts with 'answer', 'citations', 'reasoning_steps'.

    Args:
        eval_file: path to eval_cases.json
        agent_pipeline: AgenticRAGPipeline instance
        top_k: not used directly but passed for compatibility

    Returns:
        dict with answer-level metrics (no retrieval recall since agent
        abstracts away evidence IDs)
    """
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    rows: list[dict] = []
    totals: dict[str, float] = defaultdict(float)
    counters: dict[str, int] = defaultdict(int)

    for case in cases:
        result = agent_pipeline.run(
            case["query"],
            image_path=case.get("image_path"),
            dataset_hint=case.get("dataset_hint"),
        )
        pred_answer = result.get("answer", "")
        gold_answer = case.get("gold_answer")

        em = exact_match(pred_answer, gold_answer) if gold_answer else 0.0
        f1 = token_f1(pred_answer, gold_answer) if gold_answer else 0.0

        row = {
            "query": case["query"],
            "dataset": case.get("dataset"),
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "exact_match": em,
            "token_f1": f1,
            "reasoning_steps": result.get("reasoning_steps", []),
            "step_count": result.get("step_count", 0),
            "citations": result.get("citations", []),
            "error": result.get("error"),
        }
        rows.append(row)

        totals["em"] += em
        totals["f1"] += f1
        counters["total"] += 1
        if gold_answer:
            counters["ans_cases"] += 1

    total = counters["total"]
    ans_cases = counters["ans_cases"]

    return {
        "total": total,
        "exact_match": _safe_div(totals["em"], ans_cases),
        "token_f1": _safe_div(totals["f1"], ans_cases),
        "mean_steps": _safe_div(
            sum(r.get("step_count", 0) for r in rows), total
        ),
        "error_count": sum(1 for r in rows if r.get("error")),
        "per_dataset": _compute_per_dataset(rows),
        "rows": rows,
    }
