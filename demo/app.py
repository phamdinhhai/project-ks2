"""Gradio interactive demo for Medical Multimodal RAG Agent.

Usage:
  # Baseline mode (no GPU)
  python demo/app.py

  # Agent mode (GPU + Qdrant)
  python demo/app.py --use-agent --use-qdrant

  # API mode (no GPU needed)
  QWEN_API_KEY=xxx python demo/app.py --use-agent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the src package is importable when running directly
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _build_baseline_pipeline(args: argparse.Namespace):
    from medical_rag.config import RAGConfig
    from medical_rag.pipeline import MedicalRAGPipeline

    config = RAGConfig(
        index_dir=Path(args.index_dir),
        data_dir=Path(args.data_dir),
    ).resolved(_project_root)
    return MedicalRAGPipeline(config)


def _build_agent_pipeline(args: argparse.Namespace):
    from medical_rag.agents.graph import AgenticRAGPipeline

    return AgenticRAGPipeline({
        "index_dir": str((_project_root / args.index_dir).resolve()),
        "data_dir": str((_project_root / args.data_dir).resolve()),
        "use_qdrant": args.use_qdrant,
        "qdrant_url": args.qdrant_url,
        "use_vlm_generation": not args.no_vlm,
        "use_fine_grained_visual": not args.no_vlm,
        "use_cross_encoder_rerank": False,
        "text_top_k": 8,
        "image_top_k": 5,
    })


def _format_baseline_result(answer) -> tuple[str, str, str]:
    """Format MedicalRAGPipeline result for Gradio outputs."""
    evidence_md = ""
    for idx, item in enumerate(answer.evidence[:6], 1):
        mod_badge = "🖼️" if item.modality.value == "image" else "📄"
        evidence_md += (
            f"### {mod_badge} Evidence {idx} — {item.dataset}\n"
            f"**Score**: {item.fused_score:.4f} | "
            f"**ID**: `{item.id}`\n\n"
            f"> {' '.join(item.text.split())[:500]}\n\n"
            f"---\n\n"
        )

    reasoning = (
        f"- **Language**: {answer.intent.language.value}\n"
        f"- **Modality**: {answer.intent.modality}\n"
        f"- **Image branch**: {'✅' if answer.intent.use_image_branch else '❌'}\n"
        f"- **Reasons**: {', '.join(answer.intent.reasons)}\n"
    )

    return answer.answer, evidence_md, reasoning


def _format_agent_result(result: dict) -> tuple[str, str, str]:
    """Format AgenticRAGPipeline result for Gradio outputs."""
    answer = result.get("answer", "No answer generated")

    # Evidence
    text_ev = result.get("text_evidence", [])
    vis_ev = result.get("visual_evidence", [])
    evidence_md = ""
    for idx, item in enumerate(text_ev[:4] + vis_ev[:3], 1):
        mod = "🖼️" if item.get("modality") == "image" else "📄"
        evidence_md += (
            f"### {mod} Evidence {idx} — {item.get('dataset', '?')}\n"
            f"**Score**: {item.get('score', 0):.4f} | "
            f"**Source**: `{item.get('source', '?')}`\n\n"
            f"> {' '.join(item.get('text', '').split())[:500]}\n\n"
            f"---\n\n"
        )

    # Reasoning steps
    steps = result.get("reasoning_steps", [])
    reasoning = "\n".join(f"- {step}" for step in steps) if steps else "No reasoning steps"
    reasoning += f"\n\n**Total steps**: {result.get('step_count', 0)}"

    return answer, evidence_md, reasoning


def create_demo(pipeline, is_agent: bool = False):
    """Create the Gradio interface."""
    import gradio as gr

    def query_fn(question: str, image, dataset_hint: str):
        if not question.strip():
            return "Please enter a question.", "", ""

        hint = dataset_hint if dataset_hint != "auto" else None

        if is_agent:
            img_path = None
            if image is not None:
                import tempfile
                tmp = Path(tempfile.mkdtemp()) / "upload.jpg"
                image.save(str(tmp))
                img_path = str(tmp)
            result = pipeline.run(question, image_path=img_path, dataset_hint=hint)
            return _format_agent_result(result)
        else:
            answer = pipeline.run(question, dataset_hint=hint)
            return _format_baseline_result(answer)

    with gr.Blocks(
        title="Medical Multimodal RAG Agent",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown(
            "# 🏥 Medical Multimodal RAG Agent\n"
            "Ask medical questions with optional image upload. "
            f"**Mode**: {'🤖 Agent (LangGraph)' if is_agent else '📊 Baseline (BM25+TF-IDF)'}"
        )

        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Medical Question",
                    placeholder="e.g. What does this chest X-ray show?",
                    lines=2,
                )
                image_input = gr.Image(
                    label="Upload Medical Image (optional)",
                    type="pil",
                    visible=is_agent,
                )
                dataset = gr.Dropdown(
                    choices=["auto", "medqa", "bioasq", "vqa_rad", "roco", "mimic_cxr", "pathvqa"],
                    value="auto",
                    label="Dataset hint",
                )
                submit_btn = gr.Button("🔍 Search & Answer", variant="primary")

            with gr.Column(scale=3):
                answer_output = gr.Textbox(label="Answer", lines=6)
                with gr.Accordion("Retrieved Evidence", open=False):
                    evidence_output = gr.Markdown(label="Evidence")
                with gr.Accordion("Reasoning Trace", open=False):
                    reasoning_output = gr.Markdown(label="Reasoning")

        submit_btn.click(
            fn=query_fn,
            inputs=[question, image_input, dataset],
            outputs=[answer_output, evidence_output, reasoning_output],
        )

        gr.Examples(
            examples=[
                ["What are the common treatments for pneumonia?", None, "medqa"],
                ["What does this chest X-ray show?", None, "vqa_rad"],
                ["Describe the pathology findings in the image", None, "pathvqa"],
            ],
            inputs=[question, image_input, dataset],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Medical RAG Gradio Demo")
    parser.add_argument("--data-dir", default="data", help="Dataset root")
    parser.add_argument("--index-dir", default="data/processed/indexes", help="Index dir")
    parser.add_argument("--use-agent", action="store_true", help="Use LangGraph agent")
    parser.add_argument("--use-qdrant", action="store_true", help="Use Qdrant")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM generation")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    if args.use_agent:
        pipeline = _build_agent_pipeline(args)
        is_agent = True
    else:
        pipeline = _build_baseline_pipeline(args)
        is_agent = False

    demo = create_demo(pipeline, is_agent=is_agent)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
