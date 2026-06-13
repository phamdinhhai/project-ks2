# Deployment Guide — Colab Indexing + Laptop QA App

This project is designed for a practical split deployment:

| Runtime | Responsibility |
|---|---|
| Laptop | Gradio/CLI QA app, baseline debug, OpenRouter generation |
| Colab | BioMedBERT/BioCLIP/BGE indexing and evaluation |
| OpenRouter | Gemini 2.5 Flash VLM/LLM generation and grounding |
| Qdrant | Vector storage: local for debug, cloud for real workflow |

## 1. Secrets

Never commit real keys. Use environment variables or Colab Secrets.

```powershell
$env:OPENROUTER_API_KEY="your_rotated_openrouter_key"
$env:OPENROUTER_MODEL="google/gemini-2.5-flash"
$env:QDRANT_URL="https://your-cluster.cloud.qdrant.io"
$env:QDRANT_API_KEY="your_qdrant_key"
```

> [!CAUTION]
> If a key was pasted into chat or committed anywhere, revoke it and create a new key.

## 2. Laptop workflow

Run local smoke tests:

```powershell
python -m pytest tests/test_smoke.py -v
python -m medical_rag test-encoders --mock
python -m medical_rag test-qdrant --qdrant-url :memory:
```

Test OpenRouter after setting a rotated key:

```powershell
python -m medical_rag test-openrouter
```

Run the app:

```powershell
python demo/app.py --use-agent --use-qdrant
```

If Qdrant Cloud is not ready yet, run baseline mode:

```powershell
python demo/app.py
```

## 3. Cloud notebook workflow

You can use either Colab or Kaggle for heavy indexing/evaluation.

### Colab notebooks

Open notebooks in order:

1. `notebooks/colab_01_build_embeddings.ipynb`
2. `notebooks/colab_02_push_qdrant.ipynb`
3. `notebooks/colab_03_run_ablation.ipynb`

### Kaggle notebooks

Recommended if you want Kaggle GPU/runtime and Kaggle Secrets:

1. `notebooks/kaggle_01_build_embeddings.ipynb`
2. `notebooks/kaggle_02_verify_qdrant.ipynb`
3. `notebooks/kaggle_03_run_ablation.ipynb`

Detailed guide: [kaggle-workflow.md](kaggle-workflow.md)

Cloud notebooks are responsible for:

- Loading BioMedBERT and encoding text chunks
- Loading BioCLIP and encoding image/full/patch vectors
- Using BGE reranker for evaluation/ablation
- Pushing vectors to Qdrant Cloud
- Exporting benchmark JSON/Markdown files

## 4. Qdrant strategy

Use in-memory/local Qdrant for debugging:

```powershell
python -m medical_rag build-qdrant-index --qdrant-url :memory: --limit 100 --use-mock-models
```

Use Qdrant Cloud for real runs:

```powershell
python -m medical_rag test-qdrant --qdrant-url $env:QDRANT_URL --use-cloud-auth
```

Build real index on Colab:

```bash
python scripts/colab_workflow.py build-index --data-dir data --limit 1000 --recreate --no-mock
```

## 5. Evaluation outputs

Expected files:

```text
outputs/benchmark/baseline_advanced.json
outputs/benchmark/agent_openrouter.json
outputs/ablation/ablation_report.md
```

These are the main artifacts for thesis/report writing.

## 6. Recommended execution order

```text
1. Rotate/set API keys
2. Laptop: test-openrouter
3. Laptop: test-qdrant :memory:
4. Colab: build real Qdrant index
5. Laptop: run Gradio app with Qdrant Cloud
6. Colab: run evaluation and ablation
7. Copy outputs back to laptop
```
