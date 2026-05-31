# Coding Standards

## Python Style

- **Python ≥ 3.10**, type hints required on all public functions
- `from __future__ import annotations` in **every file**
- Line length: 120 characters soft limit
- Formatter: none enforced currently (recommend `ruff` or `black`)

## Import Rules

```python
# ✅ GOOD — lazy import for GPU libs (inside function)
def encode_text(self, text: str):
    import torch
    from transformers import AutoModel
    ...

# ❌ BAD — top-level import of heavy libs
import torch
from transformers import AutoModel
```

**Why**: baseline pipeline must work without torch/transformers installed.

## Module Organization

| Code type | Location | Example |
|---|---|---|
| Baseline pipeline | `src/medical_rag/` root | pipeline.py, router.py |
| Baseline retrieval | `retrieval/` | text.py, image.py, fusion.py |
| Model wrappers | `models/` | bioclip.py, biomedbert.py |
| Agent nodes | `agents/` | decomposer.py, graph.py |
| Data ingestion | `ingestion/` | chunker.py, indexer.py |
| Data tools | `data_tools/` | canonicalize.py |

## Configuration

- **All settings** go in `RAGConfig` (config.py). Never hardcode paths/params elsewhere.
- New features → add a `bool` flag to config (e.g., `use_fine_grained_visual`)
- Defaults must be backward-compatible (new flags default to `False`)

## Error Handling

```python
# ✅ GOOD — clear error with install instructions
try:
    from transformers import AutoModel
except ImportError as exc:
    raise RuntimeError("Install: pip install transformers torch") from exc

# ✅ GOOD — graceful fallback
try:
    results = _search_qdrant(...)
except Exception:
    results = _search_baseline(...)  # joblib fallback
```

## Docstrings

```python
def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode a list of texts in batches.

    Args:
        texts: list of input strings
        batch_size: number of texts per forward pass

    Returns:
        np.ndarray of shape (N, 1024)
    """
```

## Testing

- All tests in `tests/test_smoke.py` — must run offline (no GPU, no network, no Docker)
- Use `tmp_path` pytest fixture for file I/O
- New modules → add at least one import/smoke test
