from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from medical_rag.pipeline import MedicalRAGPipeline
from medical_rag.reporting import answer_to_debug_dict

DEFAULT_QUERIES = [
    {"query": "pneumonia treatment", "dataset_hint": None},
    {"query": "chest xray pneumonia", "dataset_hint": None},
    {"query": "hình ảnh viêm phổi", "dataset_hint": None},
]


def _load_queries(path: Path | None) -> list[dict[str, Any]]:
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
            return payload["queries"]
    return DEFAULT_QUERIES


def _render_html(results: list[dict[str, Any]]) -> str:
    cards = []
    for result in results:
        evidence_items = []
        for evidence in result.get("evidence", []):
            source_path = evidence.get("source_path") or ""
            source_html = f"<code>{html.escape(source_path)}</code>" if source_path else ""
            evidence_items.append(
                "<li>"
                f"<strong>#{evidence['rank']} {html.escape(evidence['dataset'])}</strong> "
                f"<span class='pill'>{html.escape(evidence['modality'])}</span> "
                f"<span class='score'>{evidence['fused_score']:.4f}</span>"
                f"<p>{html.escape(evidence.get('text_preview', ''))}</p>"
                f"{source_html}"
                "</li>"
            )
        cards.append(
            "<section class='card'>"
            f"<h2>{html.escape(result['query'])}</h2>"
            f"<p class='route'>Route: {html.escape(result['intent']['modality'])} · "
            f"Language: {html.escape(result['intent']['language'])}</p>"
            f"<pre>{html.escape(result['answer'])}</pre>"
            f"<ol>{''.join(evidence_items)}</ol>"
            "</section>"
        )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Medical Multimodal RAG Demo</title>
  <meta name="description" content="Static demo export for the Medical Multimodal RAG baseline." />
  <style>
    :root { color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; }
    body { margin: 0; background: radial-gradient(circle at top left, #1f3b73, #090b14 46%, #05060a); color: #eef4ff; }
    header { padding: 48px min(7vw, 88px) 24px; }
    h1 { font-size: clamp(32px, 5vw, 64px); margin: 0; letter-spacing: -0.05em; }
    .subtitle { max-width: 820px; color: #b7c7e8; font-size: 18px; line-height: 1.65; }
    main { display: grid; gap: 24px; padding: 20px min(7vw, 88px) 80px; }
    .card { border: 1px solid rgba(146, 181, 255, .24); border-radius: 28px; padding: 26px; background: rgba(13, 19, 38, .72); box-shadow: 0 24px 80px rgba(0,0,0,.35); backdrop-filter: blur(18px); }
    h2 { margin: 0 0 8px; font-size: 26px; }
    .route { color: #9ec5ff; }
    pre { white-space: pre-wrap; background: rgba(255,255,255,.055); border-radius: 18px; padding: 18px; line-height: 1.55; overflow-x: auto; }
    ol { padding-left: 22px; }
    li { margin: 16px 0; color: #dce8ff; }
    .pill { display: inline-block; margin-left: 8px; padding: 3px 10px; border-radius: 999px; background: linear-gradient(135deg, #21d4fd, #b721ff); color: white; font-size: 12px; }
    .score { color: #8ee6c9; margin-left: 8px; }
    code { color: #ffe29a; word-break: break-all; }
  </style>
</head>
<body>
  <header>
    <h1>Medical Multimodal RAG Demo</h1>
    <p class="subtitle">Dependency-free static export showing routing, answers, citations, and retrieved text/image evidence. raw_pdfs is excluded by current project scope.</p>
  </header>
  <main>
""" + "\n".join(cards) + """
  </main>
</body>
</html>
"""


def export_demo(
    pipeline: MedicalRAGPipeline,
    output_dir: Path,
    queries_file: Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = _load_queries(queries_file)
    results = []
    for item in queries:
        query = str(item.get("query") or item)
        answer = pipeline.run(query, dataset_hint=item.get("dataset_hint"), top_k=top_k)
        results.append(answer_to_debug_dict(answer))

    data_path = output_dir / "demo_data.json"
    html_path = output_dir / "index.html"
    data_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(results), encoding="utf-8")
    return {"output_dir": str(output_dir), "html": str(html_path), "data": str(data_path), "queries": len(results)}
