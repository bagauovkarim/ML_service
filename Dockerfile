FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
ENV UV_NO_DEV=1
RUN uv sync --frozen --no-install-project
COPY README.md ./
COPY src/ ./src/
RUN uv sync --locked
CMD ["uv", "run", "uvicorn", "bank_marketing_service.inference:app", "--host", "0.0.0.0", "--port", "8000"]