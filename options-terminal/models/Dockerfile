# Build context: options-terminal/ (the runner reads contracts/ for slots and schemas).
FROM python:3.12-slim
LABEL org.opencontainers.image.title="options-terminal-runner" \
      com.options-terminal.project="options-terminal"
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/app/models/.venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app/models
COPY models/pyproject.toml models/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY models/terminal_models ./terminal_models
COPY models/registry.yaml ./registry.yaml
RUN uv sync --frozen --no-dev
COPY contracts /app/contracts
COPY docker/runner/entrypoint.sh /usr/local/bin/runner-entrypoint
RUN useradd --system --uid 10001 runner && mkdir -p /data/snapshots && chown runner /data/snapshots
USER runner
CMD ["runner-entrypoint"]
