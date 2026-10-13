# models/

Python model framework, data adapters, runner and model library. Spec: [`../MODELS.md`](../MODELS.md).

```bash
uv sync
uv run pytest
uv run lint-imports        # M-2 import rules
uv run ruff check .
uv run python -m terminal_models.runner validate-registry
```

Status: scaffolding. `validate-registry` works; `run`, `run-due`, `diff` and `replay`, the live Hyperliquid/Deribit adapters and the sanity checks are M1.
