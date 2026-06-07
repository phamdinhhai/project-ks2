from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console

from medical_rag.ablation import run_ablation, run_agent_ablation
from medical_rag.config import RAGConfig
from medical_rag.dataset_audit import audit_data_directory
from medical_rag.demo_export import export_demo as export_demo_site
from medical_rag.evaluation import evaluate as run_evaluation
from medical_rag.evaluation_advanced import evaluate_advanced as run_evaluation_advanced
from medical_rag.evaluation_advanced import ragas_evaluate as run_ragas_evaluation
from medical_rag.indexing import build_indexes, index_stats
from medical_rag.pipeline import MedicalRAGPipeline
from medical_rag.project_status import build_status_report
from medical_rag.reporting import run_query_debug, write_query_debug
from medical_rag.eval_case_builder import build_eval_cases, write_eval_cases, write_eval_summary
from medical_rag.analysis import summarize_ablation as run_summary

app = typer.Typer(help="Medical multimodal RAG baseline CLI")
console = Console()


def _print_json(payload: dict | list, ascii_safe: bool = True) -> None:
    console.print(json.dumps(payload, ensure_ascii=ascii_safe, indent=2), markup=False, highlight=False)


@app.command("build-index")
def build_index(
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index output directory"),
    limit: int | None = typer.Option(None, help="Optional per-dataset limit for quick experiments"),
) -> None:
    bundle = build_indexes(data_dir=data_dir, index_dir=index_dir, limit_per_dataset=limit)
    _print_json({"index_dir": str(index_dir), **index_stats(bundle)})


@app.command("build-qdrant-index")
def build_qdrant_index(
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    qdrant_url: str = typer.Option(":memory:", help="Qdrant URL, or ':memory:' for local in-memory"),
    limit: int | None = typer.Option(100, help="Optional per-dataset limit; default safe for laptops"),
    recreate: bool = typer.Option(False, help="Drop and recreate Qdrant collections"),
    use_patches: bool = typer.Option(True, help="Index full image + quadrant patches"),
    use_mock_models: bool = typer.Option(True, help="Use deterministic mock embeddings for local testing"),
    use_cloud_auth: bool = typer.Option(False, help="Use QDRANT_API_KEY for Qdrant Cloud"),
) -> None:
    """Build Qdrant text/image index.

    Defaults are safe for RTX 3050/CPU laptops: in-memory Qdrant, limit=100,
    and mock embeddings. Use --no-use-mock-models on Colab/GPU.
    """
    from medical_rag.ingestion.indexer import build_qdrant_index as run_qdrant_index
    from medical_rag.ingestion.indexer import get_client
    from medical_rag.models.mock_models import MockBioCLIP, MockBioMedBERT

    if use_cloud_auth:
        from medical_rag.ingestion.qdrant_cloud import get_qdrant_cloud_client
        client = get_qdrant_cloud_client(url=qdrant_url if qdrant_url != ":memory:" else None)
    else:
        client = get_client(qdrant_url)
    if use_mock_models:
        text_encoder = MockBioMedBERT()
        image_encoder = MockBioCLIP()
    else:
        from medical_rag.models.bioclip import BioCLIPEncoder
        from medical_rag.models.biomedbert import BioMedBERTEncoder
        text_encoder = BioMedBERTEncoder()
        image_encoder = BioCLIPEncoder()

    result = run_qdrant_index(
        data_dir=data_dir,
        client=client,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        limit_per_dataset=limit,
        recreate=recreate,
        use_patches=use_patches,
    )
    _print_json({"qdrant_url": qdrant_url, "mock": use_mock_models, **result})


@app.command("audit-data")
def audit_data(
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    output_file: Path | None = typer.Option(Path("data/processed/current_dataset_audit.json"), help="Optional JSON output file"),
    sample_size: int = typer.Option(2, help="Number of sample rows/files per section"),
) -> None:
    report = audit_data_directory(data_dir, sample_size=sample_size)
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    _print_json(report, ascii_safe=True)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report_json, encoding="utf-8")


@app.command("status-report")
def status_report(
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index directory"),
    output_file: Path | None = typer.Option(None, help="Optional JSON output file"),
) -> None:
    report = build_status_report(data_dir=data_dir, index_dir=index_dir)
    _print_json(report, ascii_safe=True)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


@app.command("query")
def query(
    query_text: str = typer.Argument(..., help="Medical question/query"),
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index directory"),
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    image_path: str | None = typer.Option(None, help="Optional uploaded image path"),
    dataset_hint: str | None = typer.Option(None, help="Optional dataset hint"),
    top_k: int = typer.Option(5, help="Top-k text results before fusion"),
    save_json: Path | None = typer.Option(None, help="Optional debug JSON output file"),
) -> None:
    config = RAGConfig(index_dir=index_dir, data_dir=data_dir).resolved()
    pipeline = MedicalRAGPipeline(config)
    if save_json:
        payload = run_query_debug(pipeline, query_text, image_path=image_path, dataset_hint=dataset_hint, top_k=top_k)
        write_query_debug(payload, save_json)
        _print_json({"saved": str(save_json), "evidence": len(payload.get("evidence", []))})
        return
    answer = pipeline.run(query_text, image_path=image_path, dataset_hint=dataset_hint, top_k=top_k)
    console.print(answer.answer)


@app.command("evaluate")
def evaluate(
    eval_file: Path = typer.Option(Path("data/eval_cases.json"), help="Evaluation cases JSON"),
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index directory"),
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    top_k: int = typer.Option(5, help="Top-k evidence for metrics"),
    output_file: Path | None = typer.Option(None, help="Optional JSON output file"),
    error_analysis: Path | None = typer.Option(None, help="Optional per-row error analysis JSON output file"),
) -> None:
    config = RAGConfig(index_dir=index_dir, data_dir=data_dir).resolved()
    pipeline = MedicalRAGPipeline(config)
    metrics = run_evaluation(eval_file, pipeline, top_k=top_k)
    _print_json(metrics)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if error_analysis:
        error_analysis.parent.mkdir(parents=True, exist_ok=True)
        error_analysis.write_text(json.dumps(metrics.get("rows", []), ensure_ascii=False, indent=2), encoding="utf-8")


@app.command("ablate")
def ablate(
    eval_file: Path = typer.Option(Path("data/eval_cases.json"), help="Evaluation cases JSON"),
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index directory"),
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    output_dir: Path = typer.Option(Path("outputs/ablation"), help="Ablation output directory"),
    top_k: int = typer.Option(5, help="Top-k evidence for metrics"),
) -> None:
    summary = run_ablation(eval_file=eval_file, index_dir=index_dir, data_dir=data_dir, output_dir=output_dir, top_k=top_k)
    _print_json(summary)


@app.command("ablate-agent")
def ablate_agent(
    eval_file: Path = typer.Option(Path("data/eval_cases_smoke_10.json"), help="Evaluation cases JSON"),
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index directory"),
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    output_dir: Path = typer.Option(Path("outputs/agent_ablation"), help="Agent ablation output directory"),
    top_k: int = typer.Option(5, help="Top-k evidence for metrics"),
    profiles: str = typer.Option("E,F,G,H", help="Comma-separated agent profiles"),
    use_mock_models: bool = typer.Option(True, help="Use mock VLM/encoders for local smoke tests"),
) -> None:
    """Run LangGraph agent ablation profiles E/F/G/H."""
    profile_list = [p.strip().upper() for p in profiles.split(",") if p.strip()]
    summary = run_agent_ablation(
        eval_file=eval_file,
        index_dir=index_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        top_k=top_k,
        profiles=profile_list,
        use_mock_models=use_mock_models,
    )
    printable = {k: v for k, v in summary.items() if k != "results"}
    _print_json(printable)


@app.command("export-demo")
def export_demo(
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index directory"),
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    queries: Path | None = typer.Option(None, help="Optional JSON file with demo queries"),
    output_dir: Path = typer.Option(Path("demo"), help="Demo output directory"),
    top_k: int = typer.Option(5, help="Top-k evidence"),
) -> None:
    config = RAGConfig(index_dir=index_dir, data_dir=data_dir).resolved()
    pipeline = MedicalRAGPipeline(config)
    payload = export_demo_site(pipeline=pipeline, output_dir=output_dir, queries_file=queries, top_k=top_k)
    _print_json(payload)


@app.command("build-eval-cases")
def build_cases(
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    output_file: Path = typer.Option(Path("data/eval_cases_expanded.json"), help="Output JSON file"),
    summary_file: Path | None = typer.Option(None, help="Optional summary JSON file"),
    target_count: int = typer.Option(40, help="Approximate target cases per dataset"),
) -> None:
    targets = {
        "medqa": target_count,
        "bioasq": target_count,
        "vqa_rad": int(target_count * 1.5),
        "roco": target_count,
        "mimic_cxr": target_count,
    }
    cases, summary = build_eval_cases(data_dir, targets=targets)
    write_eval_cases(cases, output_file)
    if summary_file:
        write_eval_summary(summary, summary_file)
    _print_json(summary)


@app.command("evaluate-advanced")
def evaluate_advanced(
    eval_file: Path = typer.Option(Path("data/eval_cases.json"), help="Evaluation cases JSON"),
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index directory"),
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    top_k: int = typer.Option(5, help="Top-k evidence for metrics"),
    output_file: Path | None = typer.Option(None, help="Optional JSON output file"),
    run_ragas: bool = typer.Option(False, help="Run optional RAGAS metrics after advanced evaluation"),
    ragas_output_file: Path | None = typer.Option(None, help="Optional RAGAS JSON output file"),
    ragas_max_samples: int | None = typer.Option(None, help="Optional max samples for RAGAS"),
) -> None:
    """Advanced evaluation: Exact Match + Token F1 + per-dataset breakdown."""
    config = RAGConfig(index_dir=index_dir, data_dir=data_dir).resolved()
    pipeline = MedicalRAGPipeline(config)
    metrics = run_evaluation_advanced(eval_file, pipeline, top_k=top_k)
    if run_ragas:
        ragas_metrics = run_ragas_evaluation(metrics, max_samples=ragas_max_samples)
        metrics["ragas"] = ragas_metrics
        if ragas_output_file:
            ragas_output_file.parent.mkdir(parents=True, exist_ok=True)
            ragas_output_file.write_text(json.dumps(ragas_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            console.print(f"[green]Saved RAGAS metrics to {ragas_output_file}[/green]")
    # Print summary without per-row noise
    summary = {k: v for k, v in metrics.items() if k != "rows"}
    _print_json(summary)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]Saved to {output_file}[/green]")


@app.command("evaluate-agent")
def evaluate_agent(
    eval_file: Path = typer.Option(Path("data/eval_cases.json"), help="Evaluation cases JSON"),
    index_dir: Path = typer.Option(Path("data/processed/indexes"), help="Index directory"),
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    top_k: int = typer.Option(5, help="Top-k evidence"),
    output_file: Path | None = typer.Option(None, help="Optional JSON output file"),
    use_qdrant: bool = typer.Option(False, help="Use Qdrant for retrieval"),
    use_vlm: bool = typer.Option(False, help="Use Qwen2.5-VL/OpenRouter for generation"),
    use_mock_models: bool = typer.Option(False, help="Use deterministic mock encoders/VLM for laptop tests"),
) -> None:
    """Evaluate the LangGraph agent pipeline (EM + F1 + reasoning steps)."""
    from medical_rag.agents.graph import AgenticRAGPipeline
    from medical_rag.evaluation_advanced import evaluate_agent as run_agent_eval

    import os

    text_model_name = os.environ.get(
        "BIOMEDBERT_MODEL",
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
    )
    image_model_name = os.environ.get(
        "BIOCLIP_MODEL",
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )

    cfg_dict = {
        "index_dir": str(index_dir),
        "data_dir": str(data_dir),
        "use_qdrant": use_qdrant,
        "use_vlm_generation": use_vlm,
        "use_fine_grained_visual": use_vlm,
        "use_cross_encoder_rerank": False,
        "text_top_k": top_k,
        "image_top_k": top_k,
        "text_model_name": text_model_name,
        "image_model_name": image_model_name,
        "qdrant_url": os.environ.get("QDRANT_URL", ":memory:"),
        "qdrant_api_key": os.environ.get("QDRANT_API_KEY"),
        "use_cloud_auth": bool(os.environ.get("QDRANT_API_KEY")),
        "llm_provider": "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "auto",
        "openrouter_model": os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        "openrouter_base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "use_mock_models": use_mock_models,
    }
    agent = AgenticRAGPipeline(cfg_dict)
    metrics = run_agent_eval(eval_file, agent, top_k=top_k)
    summary = {k: v for k, v in metrics.items() if k != "rows"}
    _print_json(summary)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]Saved to {output_file}[/green]")


@app.command("summarize-ablation")
def summarize_ablation(
    ablation_dir: Path = typer.Option(Path("outputs/ablation"), help="Ablation results directory"),
    output_file: Path = typer.Option(Path("outputs/ablation/ablation_report.md"), help="Output Markdown report"),
) -> None:
    report = run_summary(ablation_dir, output_file=output_file)
    console.print(report)


@app.command("download-datasets")
def download_datasets(
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    dataset: str = typer.Option("all", help="Dataset name or 'all'"),
    profile: str = typer.Option("full", help="Download profile (quick/full)"),
    download_only: bool = typer.Option(False, help="Only download, don't canonicalize"),
) -> None:
    cmd = ["python", "scripts/download_datasets.py", "--data-dir", str(data_dir), "--dataset", dataset, "--profile", profile]
    if download_only:
        cmd.append("--download-only")
    console.print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.command("canonicalize-datasets")
def canonicalize_datasets(
    data_dir: Path = typer.Option(Path("data"), help="Dataset root directory"),
    dataset: str = typer.Option("all", help="Dataset name or 'all'"),
    limit: int | None = typer.Option(None, help="Optional row limit per split"),
) -> None:
    from medical_rag.data_tools.canonicalize import canonicalize_all, canonicalize_dataset
    if dataset == "all":
        report = canonicalize_all(data_dir, limit=limit)
    else:
        report = {"datasets": {dataset: canonicalize_dataset(dataset, data_dir, limit=limit)}}
    _print_json(report)


@app.command("test-openrouter")
def test_openrouter(
    prompt: str = typer.Option("What is pneumonia?", help="Prompt for the provider smoke test"),
    model: str | None = typer.Option(None, help="Override OPENROUTER_MODEL"),
    max_tokens: int = typer.Option(128, help="Max generation tokens"),
) -> None:
    """Smoke test OpenRouter/Gemini 2.5 Flash connectivity."""
    import os

    if not os.environ.get("OPENROUTER_API_KEY"):
        _print_json({
            "ok": False,
            "error": "OPENROUTER_API_KEY is not set. Rotate any pasted key and set a new key via env var.",
        })
        raise typer.Exit(code=1)

    from medical_rag.models.openrouter_vlm import OpenRouterVLM

    vlm = OpenRouterVLM(model=model, max_tokens=max_tokens)
    answer = vlm.generate(prompt, context="This is a connectivity smoke test.")
    _print_json({"ok": True, "model": vlm.model, "answer_preview": answer[:500]}, ascii_safe=False)


@app.command("test-qdrant")
def test_qdrant(
    qdrant_url: str = typer.Option(":memory:", help="Qdrant URL or ':memory:'"),
    use_cloud_auth: bool = typer.Option(False, help="Use QDRANT_API_KEY for Qdrant Cloud"),
) -> None:
    """Smoke test Qdrant local/in-memory/cloud connectivity."""
    if use_cloud_auth:
        from medical_rag.ingestion.qdrant_cloud import get_qdrant_cloud_client
        client = get_qdrant_cloud_client(url=qdrant_url if qdrant_url != ":memory:" else None)
    else:
        from medical_rag.ingestion.indexer import get_client
        client = get_client(qdrant_url)

    collections = client.get_collections().collections
    payload = {"ok": True, "collections": []}
    for c in collections:
        count = None
        try:
            count = client.count(collection_name=c.name).count
        except Exception:
            count = None
        payload["collections"].append({"name": c.name, "points": count})
    _print_json(payload)


@app.command("test-encoders")
def test_encoders(
    mock: bool = typer.Option(True, help="Use mock encoders; set --no-mock for real models"),
    include_bge: bool = typer.Option(True, help="Test BGE reranker or mock reranker"),
) -> None:
    """Smoke test BioMedBERT/BioCLIP/BGE wrappers or mock equivalents."""
    if mock:
        from medical_rag.models.mock_models import MockBGEReranker, MockBioCLIP, MockBioMedBERT
        text_encoder = MockBioMedBERT()
        image_encoder = MockBioCLIP()
        reranker = MockBGEReranker() if include_bge else None
    else:
        from medical_rag.models.bioclip import BioCLIPEncoder
        from medical_rag.models.biomedbert import BioMedBERTEncoder
        text_encoder = BioMedBERTEncoder()
        image_encoder = BioCLIPEncoder()
        if include_bge:
            from medical_rag.models.bge_reranker import BGEReranker
            reranker = BGEReranker()
        else:
            reranker = None

    text_vec = text_encoder.encode("pneumonia treatment")
    image_text_vec = image_encoder.encode_text("chest x-ray opacity")
    payload = {
        "ok": True,
        "mock": mock,
        "biomedbert_shape": list(text_vec.shape),
        "bioclip_text_shape": list(image_text_vec.shape),
    }
    if reranker:
        scored = reranker.rerank(
            "pneumonia treatment",
            [
                {"id": "a", "text": "Antibiotics may treat bacterial pneumonia."},
                {"id": "b", "text": "Fractures are bone injuries."},
            ],
            top_k=2,
        )
        payload["reranker_top_id"] = scored[0].id
        payload["reranker_top_score"] = scored[0].score
    _print_json(payload)


if __name__ == "__main__":
    app()
