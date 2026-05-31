# CLI Reference — All Commands

Package: `python -m medical_rag <command> [options]`

---

## `build-index`
Build joblib BM25+TF-IDF index from processed JSONL files.
```powershell
python -m medical_rag build-index --data-dir data --index-dir data/processed/indexes
python -m medical_rag build-index --data-dir data --index-dir data/processed/indexes --limit 500
```

## `audit-data`
Audit dataset directory — check JSONL structure, row counts, image paths.
```powershell
python -m medical_rag audit-data --data-dir data
python -m medical_rag audit-data --data-dir data --output-file data/processed/audit.json
```

## `status-report`
Project status report: implementation priorities + index stats + data summary.
```powershell
python -m medical_rag status-report --data-dir data --index-dir data/processed/indexes
```

## `query`
Run a single query through the baseline pipeline.
```powershell
python -m medical_rag query "pneumonia treatment" --index-dir data/processed/indexes
python -m medical_rag query "chest xray" --image-path xray.jpg --top-k 5
python -m medical_rag query "pneumonia" --save-json outputs/debug.json
```

## `evaluate`
Evaluate pipeline on eval cases file.
```powershell
python -m medical_rag evaluate --eval-file data/eval_cases.json --index-dir data/processed/indexes
python -m medical_rag evaluate --eval-file data/eval_cases.json --top-k 5 --output-file outputs/eval.json --error-analysis outputs/errors.json
```

## `ablate`
Run ablation study across profiles A/B/C/D.
```powershell
python -m medical_rag ablate --eval-file data/eval_cases.json --output-dir outputs/ablation
```

## `summarize-ablation`
Generate markdown report from ablation results.
```powershell
python -m medical_rag summarize-ablation --ablation-dir outputs/ablation --output-file outputs/ablation/report.md
```

## `build-eval-cases`
Build evaluation cases from processed JSONL.
```powershell
python -m medical_rag build-eval-cases --data-dir data --output-file data/eval_cases.json
python -m medical_rag build-eval-cases --target-count 100 --output-file data/eval_cases_large.json
```

## `export-demo`
Export static HTML demo.
```powershell
python -m medical_rag export-demo --output-dir demo --top-k 5
python -m medical_rag export-demo --queries queries.json --output-dir demo
```

## `download-datasets`
Download HuggingFace datasets.
```powershell
python -m medical_rag download-datasets --dataset all --profile quick
python -m medical_rag download-datasets --dataset vqa_rad --profile full
```

## `canonicalize-datasets`
Canonicalize local datasets to JSONL + manifest.
```powershell
python -m medical_rag canonicalize-datasets --dataset all
python -m medical_rag canonicalize-datasets --dataset medqa --limit 1000
```
