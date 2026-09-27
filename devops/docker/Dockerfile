FROM python:3.14-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY src ./src
COPY Readme.md ./

RUN uv sync --no-dev --no-editable --frozen

# LiteLLM gateway for the anthropic_sdk engine. It lives in its own venv because its
# dependencies conflict with the app's (litellm[proxy] pins mcp<2). To remove it: delete
# this stage, the COPY and LITELLM_COMMAND below, and set LITELLM_ENABLED=false.
FROM python:3.14-slim AS litellm

RUN pip install --no-cache-dir uv \
    && uv venv /opt/litellm \
    && uv pip install --python /opt/litellm/bin/python --no-cache "litellm[proxy]==1.102.1"

FROM python:3.14-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=litellm /opt/litellm /opt/litellm
COPY src ./src
COPY config ./config
RUN cp -r config/debug config/deploy

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    CONFIGURATION_DIR="/app/config" \
    LITELLM_COMMAND="/opt/litellm/bin/litellm" \
    WORKING_DIRECTORY="/data/working_directory"

RUN mkdir -p /data/sqlite /data/working_directory && chown -R app:app /data

USER app
EXPOSE 8000

CMD ["uvicorn", "bootstrap.application.agent_application:app", "--host", "0.0.0.0", "--port", "8000"]
