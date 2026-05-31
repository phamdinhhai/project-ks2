# Dataset Standardization

This project uses a manifest-driven canonical dataset layout for all non-PDF medical datasets.

## Scope

`data/raw_pdfs` is intentionally excluded for now. Active datasets:

| Dataset | Source | Canonical files | Rows |
|---|---|---|---:|
| MedQA | local JSONL | `medqa_*.canonical.jsonl` | 12,723 |
| BioASQ | HF `hf_dataset` | `bioasq_*.jsonl` | 8,216 |
| VQA-RAD | HF `hf_dataset` | `vqa_rad_*.jsonl` | 2,244 |
| ROCO | local 2.5GB subset | `roco_train.jsonl` | 12,415 |
| MIMIC-CXR | HF `hf_dataset` | `mimic_cxr_train.jsonl` | 30,633 |
| PathVQA | HF `hf_dataset` (1GB-constrained subset) | `pathvqa_*.jsonl` | 3,600 |

## Canonical Schema

Each canonical row contains:

```json
{
  "dataset": "pathvqa",
  "split": "train",
  "record_id": "...",
  "question": "...",
  "answer": "...",
  "text": "...",
  "image_path": "pathvqa/images/train/....jpg",
  "metadata": {}
}
```

Every dataset folder contains a `manifest.json` with source provenance, split counts,
canonical fields, file paths, row counts, and SHA-256 hashes.

## Commands

Download and canonicalize all datasets:

```powershell
python -m medical_rag download-datasets --data-dir data --dataset all --profile full
```

Canonicalize from already downloaded/local sources:

```powershell
python -m medical_rag canonicalize-datasets --data-dir data --dataset all
```

Audit raw canonical and processed datasets:

```powershell
python -m medical_rag audit-data --data-dir data --output-file data/processed/current_dataset_audit.json
```

Build the full index:

```powershell
python -m medical_rag build-index --data-dir data --index-dir data/processed/indexes
```

## Verification Snapshot

Latest full index build:

| Dataset | Documents | Images |
|---|---:|---:|
| BioASQ | 8,216 | 0 |
| MedQA | 12,723 | 0 |
| MIMIC-CXR | 30,633 | 30,633 |
| PathVQA | 3,600 | 3,600 |
| ROCO | 12,415 | 12,415 |
| VQA-RAD | 2,244 | 2,244 |

Total indexed objects:

- Documents: 69,831
- Images: 48,892
- ID aliases: 377,108

Smoke tests:

```powershell
pytest
# 8 passed
```

## Notes

- ROCO intentionally uses the existing 2.5GB subset only.
- PathVQA is intentionally capped to a **<=1GB subset** (current size ~850MB) for resource-safe training/indexing.
- MIMIC-CXR access still depends on local/gated source availability.
- PDF ingestion remains out of scope until explicitly re-enabled.
