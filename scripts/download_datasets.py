"""Download and canonicalize medical datasets.

Usage:
  python scripts/download_datasets.py --profile quick
  python scripts/download_datasets.py --dataset pathvqa --profile full
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from medical_rag.data_tools.canonicalize import CANONICAL_DATASETS, canonicalize_all, canonicalize_dataset

# Configuration for HuggingFace datasets
DATASET_REPOS = {
    "bioasq": "kroshan/BioASQ",
    "vqa_rad": "flaviagiammarino/vqa-rad",
    "roco": "eltorio/ROCOv2-radiology",
    "mimic_cxr": "itsanmolgupta/mimic-cxr-dataset",
    "pathvqa": "flaviagiammarino/path-vqa",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ensure_dataset_downloaded(dataset: str, data_dir: Path, limit: int | None = None) -> Path | None:
    """Download dataset from HF if hf_dataset directory is missing."""
    if dataset == "medqa":
        return data_dir / "medqa"  # MedQA is local-only in this setup

    repo_id = DATASET_REPOS.get(dataset)
    if not repo_id:
        return None

    output_dir = data_dir / dataset / "hf_dataset"
    # For ROCO, we might be using a specific subset folder
    if dataset == "roco" and (data_dir / "roco" / "hf_subset_2_5gb").exists():
        return data_dir / "roco" / "hf_subset_2_5gb"

    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info(f"Dataset {dataset} already exists at {output_dir}")
        return output_dir

    logger.info(f"Downloading {dataset} from {repo_id}...")
    try:
        from datasets import DatasetDict, load_dataset

        ds = load_dataset(repo_id)

        if limit:
            logger.info(f"Applying limit of {limit} rows per split for {dataset}")
            if hasattr(ds, "keys"):
                limited = {}
                for split in ds.keys():
                    n = min(limit, len(ds[split]))
                    limited[split] = ds[split].select(range(n))
                ds = DatasetDict(limited)
            else:
                n = min(limit, len(ds))
                ds = ds.select(range(n))

        output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_dir))

        split_counts = {}
        features = []
        if hasattr(ds, "keys"):
            split_counts = {str(split): int(len(ds[split])) for split in ds.keys()}
            first_split = next(iter(ds.keys()), None)
            if first_split is not None:
                features = list(ds[first_split].features.keys())
        else:
            split_counts = {"train": int(len(ds))}
            features = list(ds.features.keys())

        meta = {
            "status": "ok",
            "dataset_id": repo_id,
            "output_dir": str(output_dir),
            "split_counts": split_counts,
            "features": features,
            "limited_rows_per_split": limit,
        }
        meta_path = data_dir / dataset / "download_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(f"Saved {dataset} to {output_dir}")
        return output_dir
    except ImportError:
        logger.error("HuggingFace 'datasets' library not found. Run: pip install datasets")
        return None
    except Exception as e:
        logger.error(f"Failed to download {dataset}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/canonicalize medical datasets into JSONL manifests.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=["all", *CANONICAL_DATASETS], default="all")
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows per split; overrides quick profile default.")
    parser.add_argument("--download-only", action="store_true", help="Only download, don't canonicalize.")
    args = parser.parse_args()

    limit = args.limit
    if args.profile == "quick" and limit is None:
        limit = 100

    datasets_to_process = CANONICAL_DATASETS if args.dataset == "all" else [args.dataset]
    
    # Step 1: Ensure downloaded
    for ds_name in datasets_to_process:
        ensure_dataset_downloaded(ds_name, args.data_dir, limit=limit)

    if args.download_only:
        logger.info("Download only requested. Skipping canonicalization.")
        return

    # Step 2: Canonicalize
    logger.info(f"Starting canonicalization for: {', '.join(datasets_to_process)}")
    if args.dataset == "all":
        report = canonicalize_all(args.data_dir, limit=limit)
    else:
        report = {"datasets": {args.dataset: canonicalize_dataset(args.dataset, args.data_dir, limit=limit)}}
    
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
