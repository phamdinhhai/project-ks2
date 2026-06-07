"""Colab workflow helper for Medical Multimodal RAG.

This script intentionally contains no secrets. Configure secrets via Colab Secrets
or environment variables:

    OPENROUTER_API_KEY
    OPENROUTER_MODEL=google/gemini-2.5-flash
    QDRANT_URL
    QDRANT_API_KEY

Examples:
    python scripts/colab_workflow.py test-env
    python scripts/colab_workflow.py build-index --data-dir data --qdrant-url $QDRANT_URL --no-mock
    python scripts/colab_workflow.py build-index-resumable --data-dir data --datasets all --modality both --image-mode full_only --max-minutes 100
    python scripts/colab_workflow.py eval-agent --eval-file data/eval_cases.json --output-file outputs/benchmark/agent.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def cmd_test_env(_: argparse.Namespace) -> None:
    payload = {
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "openrouter_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
        "openrouter_model": os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        "qdrant_url_set": bool(os.environ.get("QDRANT_URL")),
        "qdrant_key_set": bool(os.environ.get("QDRANT_API_KEY")),
    }
    print(json.dumps(payload, indent=2))


def cmd_build_index(args: argparse.Namespace) -> None:
    qdrant_url = args.qdrant_url or os.environ.get("QDRANT_URL", ":memory:")
    cmd = [
        sys.executable,
        "-m",
        "medical_rag",
        "build-qdrant-index",
        "--data-dir",
        args.data_dir,
        "--qdrant-url",
        qdrant_url,
        "--limit",
        str(args.limit),
    ]
    if args.recreate:
        cmd.append("--recreate")
    if not args.use_patches:
        cmd.append("--no-use-patches")
    if not args.mock:
        cmd.append("--no-use-mock-models")
    _run(cmd)


def cmd_build_index_resumable(args: argparse.Namespace) -> None:
    qdrant_url = args.qdrant_url or os.environ.get("QDRANT_URL", ":memory:")
    state_file = args.state_file or os.environ.get(
        "QDRANT_INDEX_STATE",
        "outputs/index_state/full_index_state.json",
    )
    cmd = [
        sys.executable,
        "-m",
        "medical_rag",
        "build-qdrant-index-resumable",
        "--data-dir",
        args.data_dir,
        "--qdrant-url",
        qdrant_url,
        "--datasets",
        args.datasets,
        "--modality",
        args.modality,
        "--image-mode",
        args.image_mode,
        "--state-file",
        state_file,
        "--text-collection",
        args.text_collection,
        "--image-collection",
        args.image_collection,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.max_records is not None:
        cmd.extend(["--max-records", str(args.max_records)])
    if args.max_text_points is not None:
        cmd.extend(["--max-text-points", str(args.max_text_points)])
    if args.max_image_points is not None:
        cmd.extend(["--max-image-points", str(args.max_image_points)])
    if args.max_minutes is not None:
        cmd.extend(["--max-minutes", str(args.max_minutes)])
    if args.recreate:
        cmd.append("--recreate")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.mock:
        cmd.append("--use-mock-models")
    if args.use_cloud_auth:
        cmd.append("--use-cloud-auth")
    _run(cmd)

def cmd_eval_agent(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "medical_rag",
        "evaluate-agent",
        "--eval-file",
        args.eval_file,
        "--output-file",
        args.output_file,
        "--use-vlm",
    ]
    if args.use_qdrant:
        cmd.append("--use-qdrant")
    _run(cmd)


def cmd_eval_baseline(args: argparse.Namespace) -> None:
    _run([
        sys.executable,
        "-m",
        "medical_rag",
        "evaluate-advanced",
        "--eval-file",
        args.eval_file,
        "--output-file",
        args.output_file,
    ])


def cmd_ablation(args: argparse.Namespace) -> None:
    _run([
        sys.executable,
        "-m",
        "medical_rag",
        "ablate",
        "--eval-file",
        args.eval_file,
        "--output-dir",
        args.output_dir,
    ])
    _run([
        sys.executable,
        "-m",
        "medical_rag",
        "summarize-ablation",
        "--ablation-dir",
        args.output_dir,
        "--output-file",
        str(Path(args.output_dir) / "ablation_report.md"),
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab workflow helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("test-env")
    p.set_defaults(func=cmd_test_env)

    p = sub.add_parser("build-index")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--recreate", action="store_true")
    p.add_argument("--mock", action="store_true", default=False)
    p.add_argument("--no-mock", dest="mock", action="store_false")
    p.add_argument("--use-patches", action="store_true", default=True)
    p.add_argument("--no-use-patches", dest="use_patches", action="store_false")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("build-index-resumable")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--datasets", default="all")
    p.add_argument("--modality", choices=["text", "image", "both"], default="both")
    p.add_argument("--image-mode", choices=["full_only", "patches", "caption_fallback"], default="full_only")
    p.add_argument("--state-file", default=None)
    p.add_argument("--text-collection", default=os.environ.get("QDRANT_TEXT_COLLECTION", "text_chunks_prod"))
    p.add_argument("--image-collection", default=os.environ.get("QDRANT_IMAGE_COLLECTION", "image_patches_prod"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--max-text-points", type=int, default=None)
    p.add_argument("--max-image-points", type=int, default=None)
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--recreate", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock", action="store_true", default=False)
    p.add_argument("--use-cloud-auth", action="store_true", default=True)
    p.add_argument("--no-use-cloud-auth", dest="use_cloud_auth", action="store_false")
    p.set_defaults(func=cmd_build_index_resumable)

    p = sub.add_parser("eval-agent")
    p.add_argument("--eval-file", default="data/eval_cases.json")
    p.add_argument("--output-file", default="outputs/benchmark/agent_openrouter.json")
    p.add_argument("--use-qdrant", action="store_true")
    p.set_defaults(func=cmd_eval_agent)

    p = sub.add_parser("eval-baseline")
    p.add_argument("--eval-file", default="data/eval_cases.json")
    p.add_argument("--output-file", default="outputs/benchmark/baseline_advanced.json")
    p.set_defaults(func=cmd_eval_baseline)

    p = sub.add_parser("ablation")
    p.add_argument("--eval-file", default="data/eval_cases.json")
    p.add_argument("--output-dir", default="outputs/ablation")
    p.set_defaults(func=cmd_ablation)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
