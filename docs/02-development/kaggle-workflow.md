# Kaggle Workflow Guide

This guide explains how to run the heavy Medical Multimodal RAG workflow on Kaggle instead of Colab.

## 1. Recommended Architecture

| Runtime | Responsibility |
|---|---|
| Kaggle | BioMedBERT/BioCLIP indexing, Qdrant upload, evaluation, ablation |
| Laptop | Local development, CLI/Gradio QA app, lightweight tests |
| Qdrant Cloud | Persistent vector database shared by Kaggle and laptop |
| OpenRouter | Gemini 2.5 Flash generation and VLM-style grounding |

## 2. Kaggle Notebook Settings

Create a new Kaggle Notebook with:

```text
Accelerator: GPU T4/P100 or better for indexing
Internet: On
Persistence: Files only if you need downloadable outputs
```

> [!IMPORTANT]
> Internet must be enabled. The workflow needs GitHub clone, pip installs,
> HuggingFace model downloads, Qdrant Cloud, and OpenRouter.

## 3. Kaggle Secrets

Open the notebook sidebar:

```text
Add-ons -> Secrets
```

Add these secrets:

| Secret Name | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Gemini 2.5 Flash generation/evaluation |
| `QDRANT_URL` | Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | Qdrant Cloud API key |

The Kaggle notebooks load them explicitly as:

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OPENROUTER_API_KEY")
secret_value_1 = user_secrets.get_secret("QDRANT_API_KEY")
secret_value_2 = user_secrets.get_secret("QDRANT_URL")
```

The notebooks set defaults for:

```text
OPENROUTER_MODEL=google/gemini-2.5-flash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
QDRANT_TEXT_COLLECTION=text_chunks_prod
QDRANT_IMAGE_COLLECTION=image_patches_prod
```

## 4. Kaggle Dataset Input

Attach the uploaded dataset before running any Kaggle notebook:

```text
Add Input -> oopclone989876/ks-project2-data
```

Kaggle may mount it at either path:

```text
/kaggle/input/ks-project2-data
/kaggle/input/datasets/oopclone989876/ks-project2-data
```

The Kaggle notebooks auto-detect common layouts:

```text
/kaggle/input/ks-project2-data/*
/kaggle/input/ks-project2-data/data/*
/kaggle/input/ks-project2-data/KS_Project_2/data/*
/kaggle/input/datasets/oopclone989876/ks-project2-data/*
/kaggle/input/datasets/oopclone989876/ks-project2-data/data/*
```

They also handle a single `.zip` file if the dataset was uploaded as a zip.

## 5. Notebook Order

Run in this order:

```text
notebooks/kaggle_01_build_embeddings.ipynb
notebooks/kaggle_02_verify_qdrant.ipynb
notebooks/kaggle_03_run_ablation.ipynb
```

### Before-running checklist

> [!IMPORTANT]
> Confirm these items before running heavy cells:
>
> - Kaggle Internet is On.
> - GPU accelerator is enabled for embedding/indexing runs.
> - `OPENROUTER_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` secrets exist.
> - Qdrant collection names point to the intended environment.
> - You are running the canonical notebooks above, not a local copy notebook.

Recommended progression:

| Run type | Purpose | Safe defaults |
|---|---|---|
| Dry run | Verify imports/secrets/paths | `--max-records 10`, no recreate |
| Small run | Validate Qdrant writes | `--max-records 1000`, no recreate |
| Resume run | Continue production indexing | remove/increase record limit, no recreate |
| Rebuild run | Intentionally clear and rebuild collections | use `--recreate` only after backup/confirmation |

> [!CAUTION]
> `--recreate` deletes existing Qdrant collections before rebuilding. Do not use it for normal resume runs.

## 5. Notebook 01: Build Embeddings

This notebook:

1. Clones the GitHub repo to `/kaggle/working/project-ks2`.
2. Installs GPU/Qdrant/agent/eval dependencies.
3. Loads Kaggle Secrets into environment variables.
4. Verifies GPU and Qdrant Cloud connectivity.
5. Optionally tests BioMedBERT/BioCLIP/BGE imports.
6. Runs a dry run.
7. Runs a resumable real indexing job.
8. Copies output state files to `/kaggle/working/artifacts`.

Main command:

```bash
python scripts/colab_workflow.py build-index-resumable \
  --data-dir data \
  --qdrant-url "$QDRANT_URL" \
  --datasets all \
  --modality both \
  --image-mode full_only \
  --max-records 1000 \
  --max-minutes 100 \
  --batch-size 16 \
  --use-cloud-auth
```

### Scaling up

After the first successful run, increase or remove:

```text
--max-records 1000
--max-minutes 100
```

Recommended larger run:

```bash
python scripts/colab_workflow.py build-index-resumable \
  --data-dir data \
  --qdrant-url "$QDRANT_URL" \
  --datasets all \
  --modality both \
  --image-mode full_only \
  --max-minutes 540 \
  --batch-size 16 \
  --use-cloud-auth
```


### Data audit note

The audit cell intentionally runs:

```bash
python -m medical_rag audit-data --data-dir data
python -m medical_rag status-report --data-dir data
```

`project-status` is not a valid CLI command. If you see processed image paths with
`image_path_existing: 0` but canonical raw files have existing image paths, continue
with the dry run; loaders can normalize/copy paths during indexing.

## 6. Notebook 02: Verify Qdrant

This notebook verifies:

- secrets are loaded,
- Qdrant Cloud is reachable,
- collections exist,
- point counts are visible.

Use it after each indexing session.

## 7. Notebook 03: Evaluation / Ablation

This notebook:

1. Tests OpenRouter.
2. Tests Qdrant Cloud.
3. Runs baseline metrics.
4. Runs agent metrics with Qdrant Cloud + OpenRouter.
5. Runs ablation summary.
6. Copies outputs to `/kaggle/working/artifacts`.

Expected outputs:

```text
/kaggle/working/project-ks2/outputs/benchmark/baseline_advanced.json
/kaggle/working/project-ks2/outputs/benchmark/agent_openrouter.json
/kaggle/working/project-ks2/outputs/ablation/ablation_report.md
/kaggle/working/artifacts/outputs/...
```

## 8. Downloading Results

After a Kaggle run finishes, open the notebook right sidebar:

```text
Output -> /kaggle/working/artifacts
```

Download the generated JSON/Markdown files for thesis/report writing.

## 9. Laptop After Kaggle Indexing

Once Kaggle has pushed vectors to Qdrant Cloud, your laptop can query the same index.

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="..."
$env:OPENROUTER_MODEL="google/gemini-2.5-flash"
$env:QDRANT_URL="https://your-cluster.cloud.qdrant.io"
$env:QDRANT_API_KEY="..."

python -m medical_rag test-openrouter
python -m medical_rag test-qdrant --qdrant-url $env:QDRANT_URL --use-cloud-auth
python demo/app.py --use-agent --use-qdrant
```

## 10. Troubleshooting

### `git clone` fails

Enable Kaggle Internet:

```text
Notebook Settings -> Internet -> On
```

### Kaggle Secrets return empty values

Confirm the secret names are exactly:

```text
OPENROUTER_API_KEY
QDRANT_URL
QDRANT_API_KEY
```

### CUDA unavailable

Enable a GPU accelerator. For evaluation-only runs, CPU is acceptable.

### Model download is slow

This is expected on first run. Kaggle sessions are ephemeral, so models may need to re-download.

### Qdrant has old mock vectors

Use the real indexing notebook with `--recreate` only when you intentionally want to clear collections. Otherwise use the resumable indexer with production collection names.

> [!CAUTION]
> `--recreate` deletes existing Qdrant collections before rebuilding. Use it only when you are sure.
