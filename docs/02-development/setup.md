# Dev Environment Setup

## Prerequisites
- Python ≥ 3.10
- Git
- (Optional) CUDA GPU ≥ 16GB
- (Optional) Docker Desktop

## Quick Setup (baseline, no GPU)

```powershell
git clone <repo-url>
cd KS_Project_2

# Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package in editable mode
python -m pip install -e ".[test]"

# Verify
python -m pytest tests/test_smoke.py -v
```

## Full Setup (GPU + Qdrant + Agent)

```powershell
# Install all dependencies
python -m pip install -e ".[all]"

# Start Qdrant
docker run -d -p 6333:6333 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify Qdrant
python -c "from qdrant_client import QdrantClient; c=QdrantClient('http://localhost:6333'); print(c.get_collections())"
```

## API-only Mode (no GPU)

Set environment variable for DashScope API:
```powershell
$env:QWEN_API_KEY = "your-dashscope-key"
```

## Data Setup

Data should already be in `data/`. If not:
```powershell
python -m medical_rag download-datasets --dataset all --profile quick
python -m medical_rag build-index --data-dir data --index-dir data/processed/indexes
```
