# Contributing

## Coding Standards

- **Python ≥ 3.10** with type hints
- **`from __future__ import annotations`** in every file
- **Lazy imports** for GPU/model libraries (inside functions, not top-level)
- **Pydantic v2** for data models, **typer** for CLI
- Docstrings for all public functions/classes

## Commit Convention

```
feat: add BioCLIP encoder wrapper
fix: handle missing image_path in VQA-RAD loader
docs: update README with agent pipeline usage
test: add smoke test for visual retriever fallback
refactor: extract _search_qdrant from retrieve_text
```

## Branch Strategy

- `main` — stable, tests pass
- `dev` — active development
- `feature/<name>` — feature branches

## Pull Request Checklist

- [ ] All 8+ smoke tests pass: `pytest tests/test_smoke.py`
- [ ] No baseline behavior broken
- [ ] New code has docstrings
- [ ] Config changes go in `RAGConfig` only
- [ ] Model loading is lazy (no top-level import of torch/transformers)

## AI Coding Guidelines

When using AI to write code for this project, always provide `AI_GUIDELINES.md` as context first. Key rules:

1. Read `AI_GUIDELINES.md` before making any changes
2. Never break `MedicalRAGPipeline.run()` interface
3. All model code goes in `src/medical_rag/models/`
4. Agent code goes in `src/medical_rag/agents/`
5. Lazy import GPU libraries
6. Add fallbacks for missing dependencies
