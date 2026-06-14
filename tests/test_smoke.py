import json
from pathlib import Path

from medical_rag.ablation import run_ablation
from medical_rag.analysis import summarize_ablation
from medical_rag.config import RAGConfig
from medical_rag.data_loaders import load_corpora
from medical_rag.dataset_audit import audit_data_directory
from medical_rag.demo_export import export_demo
from medical_rag.eval_case_builder import build_eval_cases, write_eval_cases
from medical_rag.evaluation import evaluate
from medical_rag.evaluation_advanced import evaluate_advanced
from medical_rag.indexing import build_indexes
from medical_rag.pipeline import MedicalRAGPipeline
from medical_rag.project_status import build_status_report
from medical_rag.reporting import run_query_debug
from medical_rag.router import QueryRouter
from medical_rag.schema import Modality
from scripts.analyze_retrieval_errors import build_error_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _create_processed_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    image_dir = data_dir / "vqa_rad" / "images" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "train_0.jpg"
    image_path.write_bytes(b"fake-image")

    _write_jsonl(processed_dir / "medqa_processed.jsonl", [{
        "dataset": "medqa",
        "split": "train",
        "source_id": "fixture/medqa",
        "record_id": "train-0",
        "text": "Pneumonia treatment is antibiotics when bacterial infection is suspected.",
        "question": "What treats pneumonia?",
        "answer": "Antibiotics",
        "image_path": None,
        "metadata": {"raw": {}},
    }])
    _write_jsonl(processed_dir / "bioasq_processed.jsonl", [{
        "dataset": "bioasq",
        "split": "train",
        "source_id": "fixture/bioasq",
        "record_id": "train-0",
        "text": "Tuberculosis symptoms include cough and fever.",
        "question": "What are tuberculosis symptoms?",
        "answer": "cough and fever",
        "image_path": None,
        "metadata": {"raw": {}},
    }])
    _write_jsonl(processed_dir / "vqa_rad_processed.jsonl", [{
        "dataset": "vqa_rad",
        "split": "train",
        "source_id": "fixture/vqa-rad",
        "record_id": "train-0",
        "text": "Chest xray shows pneumonia opacity.",
        "question": "What does the chest xray show?",
        "answer": "Pneumonia opacity",
        "image_path": str(image_path),
        "metadata": {"raw": {}},
    }])
    _write_jsonl(processed_dir / "roco_processed.jsonl", [{
        "dataset": "roco",
        "split": "main",
        "source_id": "fixture/roco",
        "record_id": "main-0",
        "text": "Chest CT image showing pneumonia.",
        "question": None,
        "answer": None,
        "image_path": str(image_path),
        "metadata": {"raw": {}},
    }])
    _write_jsonl(processed_dir / "mimic_cxr_processed.jsonl", [{
        "dataset": "mimic_cxr",
        "split": "main",
        "source_id": "fixture/mimic",
        "record_id": "main-0",
        "text": "Frontal chest radiograph demonstrates pneumonia opacity.",
        "question": None,
        "answer": "pneumonia opacity",
        "image_path": str(image_path),
        "metadata": {"raw": {}},
    }])
    return data_dir


def _pipeline(tmp_path: Path) -> tuple[Path, Path, MedicalRAGPipeline]:
    data_dir = _create_processed_fixture(tmp_path)
    index_dir = tmp_path / "indexes"
    bundle = build_indexes(data_dir, index_dir, limit_per_dataset=10)
    pipeline = MedicalRAGPipeline(RAGConfig(data_dir=data_dir, index_dir=index_dir), index_bundle=bundle)
    return data_dir, index_dir, pipeline


def test_router_vietnamese_image_query():
    intent = QueryRouter().route("hình ảnh viêm phổi")
    assert intent.language.value == "vi"
    assert intent.use_image_branch is True
    assert intent.modality == Modality.IMAGE


def test_processed_loader_preferred(tmp_path: Path):
    data_dir = _create_processed_fixture(tmp_path)
    raw_medqa = data_dir / "medqa" / "medqa_train.jsonl"
    raw_medqa.parent.mkdir(parents=True, exist_ok=True)
    raw_medqa.write_text('{"question":"raw only","answer":"raw"}\n', encoding="utf-8")

    docs, images = load_corpora(data_dir, limit_per_dataset=10)

    assert any(doc.id == "medqa-text-train-0" for doc in docs)
    assert not any("raw only" in doc.text for doc in docs)
    assert any(image.id == "vqa_rad-image-train-0" for image in images)


def test_dataset_audit_summary(tmp_path: Path):
    data_dir = _create_processed_fixture(tmp_path)
    report = audit_data_directory(data_dir)

    assert report["datasets"]["medqa"]["processed"]["rows"] == 1
    assert report["datasets"]["vqa_rad"]["processed"]["image_path_existing"] == 1
    assert report["raw_pdfs"]["status"] == "excluded"
    assert report["raw_pdfs"]["used_for_indexing"] is False


def test_eval_case_builder_creates_real_ids(tmp_path: Path):
    data_dir = _create_processed_fixture(tmp_path)
    cases, summary = build_eval_cases(data_dir, targets={"medqa": 1, "bioasq": 1, "vqa_rad": 2, "roco": 1, "mimic_cxr": 1})

    ids = {case.get("gold_text_id") or case.get("gold_image_id") for case in cases}
    assert "medqa-text-medqa-train-0" in ids
    assert "vqa_rad-image-vqa_rad-train-0" in ids
    assert summary["raw_pdfs_used"] is False
    assert summary["total_cases"] == len(cases)


def test_pipeline_image_query_keeps_image_evidence(tmp_path: Path):
    _, _, pipeline = _pipeline(tmp_path)

    answer = pipeline.run("chest xray pneumonia", dataset_hint="vqa_rad")

    assert answer.intent.use_image_branch is True
    assert any(item.modality == Modality.IMAGE for item in answer.evidence)


def test_evaluation_flexible_alias_and_answer_metrics(tmp_path: Path):
    _, _, pipeline = _pipeline(tmp_path)
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(json.dumps([{
        "query": "pneumonia treatment",
        "gold_text_id": "text-0",
        "gold_answer": "Antibiotics",
        "modality": "text",
    }]), encoding="utf-8")

    metrics = evaluate(eval_file, pipeline, top_k=5)

    assert metrics["recall_at_k"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0
    assert "answer_accuracy" in metrics
    assert "error_distribution" in metrics


def test_status_report_and_query_debug(tmp_path: Path):
    data_dir, index_dir, pipeline = _pipeline(tmp_path)

    report = build_status_report(data_dir=data_dir, index_dir=index_dir)
    debug = run_query_debug(pipeline, "pneumonia treatment", top_k=3)

    assert report["scope"]["raw_pdfs_used_for_indexing"] is False
    assert len(report["implementation_priorities"]) == 6
    assert debug["query"] == "pneumonia treatment"
    assert debug["evidence"]


def test_ablation_demo_and_report_export(tmp_path: Path):
    data_dir, index_dir, pipeline = _pipeline(tmp_path)
    cases, _ = build_eval_cases(data_dir, targets={"medqa": 1, "bioasq": 1, "vqa_rad": 2, "roco": 1, "mimic_cxr": 1})
    eval_file = tmp_path / "eval.json"
    write_eval_cases(cases, eval_file)

    ablation_dir = tmp_path / "ablation"
    summary = run_ablation(eval_file, index_dir=index_dir, data_dir=data_dir, output_dir=ablation_dir, top_k=5)
    report = summarize_ablation(ablation_dir, output_file=ablation_dir / "ablation_report.md")
    demo = export_demo(pipeline, output_dir=tmp_path / "demo", top_k=3)

    assert {"A", "B", "C", "D"}.issubset(summary["results"].keys())
    assert (ablation_dir / "config_a.json").exists()
    assert "Summary Table" in report
    assert Path(demo["html"]).exists()
    assert Path(demo["data"]).exists()


def test_advanced_eval_quality_metrics_and_error_report(tmp_path: Path):
    _, _, pipeline = _pipeline(tmp_path)
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(json.dumps([
        {
            "query": "pneumonia treatment",
            "gold_text_id": "medqa-text-train-0",
            "gold_answer": "Antibiotics",
            "modality": "text",
            "dataset_hint": "medqa",
        },
        {
            "query": "unknown rare disease",
            "gold_text_id": "missing-id",
            "gold_answer": "unknown",
            "modality": "text",
            "dataset_hint": "bioasq",
        },
    ]), encoding="utf-8")

    metrics = evaluate_advanced(eval_file, pipeline, top_k=3)
    report = build_error_report(metrics, max_examples=5)

    assert "answer_non_empty_rate" in metrics
    assert "citation_coverage_rate" in metrics
    assert "groundedness_proxy_rate" in metrics
    assert metrics["answer_non_empty_rate"] > 0
    assert "Retrieval Error Analysis" in report
    assert "Top Miss Examples" in report
