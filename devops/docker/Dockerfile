FROM python:3.14-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock* ./
COPY src ./src
COPY Readme.md ./

RUN uv sync --no-dev --no-editable

FROM python:3.14-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /build/.venv /app/.venv
COPY src ./src
COPY config ./config

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    CONFIGURATION_DIR="/app/config"

USER app
EXPOSE 8000

CMD ["uvicorn", "bootstrap.application.agent_application:app", "--host", "0.0.0.0", "--port", "8000"]
