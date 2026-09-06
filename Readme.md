# agent-orchestrator

An analytics agent that talks to its MCP tool server over HTTP. The toolbox
is a separate, independently deployed service (see the `agent_toolbox`
repo) -- this repo doesn't know or care whether it's reaching that repo's
service or any other MCP server, as long as one is listening at
`TOOLBOX_URL`.

---

## Setup

```bash
uv sync --group dev
```

Copy `.env.example` to `.env` and fill in the paths/URLs:

```bash
cp .env.example .env
```

```bash
APP_ENV=debug
CHECKPOINT_DB_HOST=/absolute/path/to/sqlite
CHECKPOINT_DB_NAME=checkpoint
TOOLBOX_URL=http://localhost:8001/mcp

# optional -- default shown
# LLM_BASE_URL=http://nautilus:1234/v1
```

---

## Running it

A toolbox must be reachable at `TOOLBOX_URL` before this starts -- the agent
discovers its tools from it at boot and refuses to start if it can't be
reached. Locally that's usually the `agent_toolbox` repo running on
`:8001`; in Kubernetes it's whatever Service fronts that pod.

```bash
make run
# uv run uvicorn bootstrap.application.agent_application:app --host 0.0.0.0 --port 8000
```

| | URL |
|---|---|
| Chat (SSE) | `POST http://localhost:8000/api/v1/agent/stream` |
| Health | `http://localhost:8000/actuator/health` |

`POST /api/v1/agent/stream` body:

```json
{ "message": "how many users signed up today?", "model_name": "gpt-oss-20b", "request_id": "any-string" }
```

---

## Configuration

Everything lives under `config/` -- `config/root.yml` plus
`config/debug/connector/*.yml` and `config/debug/operation/*.yml`.

### Using an external MCP server instead of the bundled toolbox

The agent doesn't know or care whether `TOOLBOX_URL` points at the
`agent_toolbox` service or someone else's MCP server -- it's the same
client either way. Set `TOOLBOX_URL` in `.env`, or edit `base_url` in
`config/debug/connector/mcp.yml` directly.

### Using more than one MCP server at once

Add a second connector under `connector:` in `config/debug/connector/mcp.yml`,
with a name starting with `external_mcp_`:

```yaml
connector:
  external_mcp_analytics:
    name: external_mcp_analytics
    type: mcp
    base_url: https://some-other-server.example.com/mcp
    timeout: 30
    transport: streamable_http
    auth:
      type: token
      key_name: Authorization
      key_value: ${oc.env:ANALYTICS_MCP_TOKEN,''}
```

List it in `config/root.yml` next to the existing `mcp:` entries:

```yaml
mcp:
  toolbox: ${connector.toolbox}
  external_mcp_analytics: ${connector.external_mcp_analytics}
```

Restart the agent. Any connector name prefixed `external_mcp_` is picked up
automatically at boot and merged into the same tool catalogue as the
toolbox -- no code change. The boot log confirms what was found, per server:

```
Discovered 2 MCP tools from 'toolbox': users_tables, python_executor
Discovered 5 MCP tools from 'external_mcp_analytics': ...
```

---

## Development

```bash
make check       # lint + typecheck + tests
make test
make lint
make format
make typecheck
```

Live, real-LLM-and-real-toolbox quality evaluation (excluded from `make
test` by default):

```bash
make evaluation
```

### Pre-commit hooks

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pre-commit run --all-files --hook-stage pre-commit
```

---

## Deploying

```bash
make docker_build
helm install agent-orchestrator devops/helm -f devops/helm/values.yaml
```
