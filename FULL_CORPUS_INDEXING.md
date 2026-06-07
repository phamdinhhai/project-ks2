# Full-Corpus Resumable Qdrant Indexing

This guide describes how to load the full processed corpus into Qdrant Cloud
across multiple Colab sessions without restarting from zero.

## Why resumable indexing is needed

Colab Free runtimes can stop before the full corpus is embedded and uploaded.
The production indexer therefore uses:

- deterministic UUIDv5 point IDs,
- a JSON state checkpoint,
- batch-level state flushing,
- optional time and point budgets,
- idempotent Qdrant upserts.

## Current corpus scope

Processed local corpus before PathVQA:

| Dataset | Rows | Images |
|---|---:|---:|
| MedQA | 12,723 | 0 |
| BioASQ | 8,216 | 0 |
| VQA-RAD | 2,244 | 2,244 |
| ROCO | 12,415 | 12,415 |
| MIMIC-CXR | 30,633 | 30,633 |

PathVQA can now be processed into:

```text
data/processed/pathvqa_processed.jsonl
```

## Recommended production collections

Use fresh production collections instead of mixing with benchmark subset data:

```text
text_chunks_prod
image_patches_prod
```

## Step 1: Process PathVQA

```bash
python scripts/process_pathvqa.py --data-dir data
```

Validate only:

```bash
python scripts/process_pathvqa.py --data-dir data --validate-only
```

## Step 2: Dry-run counts

```bash
python -m medical_rag build-qdrant-index-resumable \
  --data-dir data \
  --qdrant-url :memory: \
  --datasets all \
  --modality both \
  --image-mode full_only \
  --dry-run \
  --use-mock-models
```

## Step 3: Colab production indexing

Text first:

```bash
python scripts/colab_workflow.py build-index-resumable \
  --data-dir data \
  --datasets all \
  --modality text \
  --image-mode full_only \
  --max-minutes 100
```

Images next:

```bash
python scripts/colab_workflow.py build-index-resumable \
  --data-dir data \
  --datasets all \
  --modality image \
  --image-mode full_only \
  --max-minutes 100
```

Both in one session if there is enough time:

```bash
python scripts/colab_workflow.py build-index-resumable \
  --data-dir data \
  --datasets all \
  --modality both \
  --image-mode full_only \
  --max-minutes 100
```

## Resume behavior

The default state path is:

```text
outputs/index_state/full_index_state.json
```

If Colab stops, rerun the same command. Already completed deterministic source
IDs are skipped according to the state file.

For Drive-backed persistence, set:

```bash
export QDRANT_INDEX_STATE=/content/drive/MyDrive/ks2/outputs/index_state/full_index_state.json
```

or set it in the notebook environment cell.

## Storage note

Start with:

```text
--image-mode full_only
```

This creates one BioCLIP vector per image. Patch mode creates up to 5 vectors per
image and can be much larger:

```text
--image-mode patches
```

Use patch mode only after validating Qdrant storage capacity.

## Small local test

```bash
python -m medical_rag build-qdrant-index-resumable \
  --data-dir data \
  --qdrant-url :memory: \
  --datasets vqa_rad,pathvqa \
  --modality both \
  --image-mode full_only \
  --max-records 5 \
  --use-mock-models \
  --state-file outputs/index_state/test_state.json
```

Run it twice. The second run should skip completed IDs.
