# agent-orchestrator

An analytics agent served over HTTP. You send it a message; it works out what you
want, proposes a plan when the request needs data or tools, runs its tools once you
approve the plan, checks its own answer, and streams the reply back as server-sent
events.

Its tools come from one or more MCP servers reached over HTTP: by default the
`agent_toolbox` service, but any MCP server works.

---

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible model server (llama.cpp, vLLM, LM Studio…) at `LLM_BASE_URL`
- Optional: an MCP server at `TOOLBOX_URL` (the agent still starts without one)
- Conversation storage: MongoDB, or a local SQLite file (the `langgraph` engine falls back
  to SQLite when MongoDB is unreachable; the `anthropic_sdk` engine needs MongoDB)

---

## Setup

```bash
uv sync --group dev
cp .env.example .env      # then fill in the URLs and paths
```

---

## Running it

```bash
make run        # LangGraph engine (default)
make run_sdk    # Claude Agent SDK engine
```

| | URL |
|---|---|
| Chat (SSE) | `POST http://localhost:8000/v1/stream` |
| Health | `GET http://localhost:8000/actuator/health` |

The boot log lists the tools found on each MCP server. A server that is down or has no
tools is skipped with a warning; its tools appear after a restart.

---

## Using the agent

### The request

```json
{ "message": "how many users signed up today?", "model_name": "qwen3-8b", "request_id": "conversation-id" }
```

| Field | Meaning |
|---|---|
| `message` | what you say (non-empty) |
| `model_name` | the model that answers; one of the models in `config/<env>/operation/llm.yml` |
| `request_id` | the conversation id. Send the same value for every message of one conversation; a new value starts a new, empty conversation |

You can change `model_name` from one message to the next in the same conversation.

### The response

A server-sent event stream. Each frame is `event: <type>` + `data: <json>`:

| Event | What to do with it |
|---|---|
| `status` | progress ("Understanding your request", "Preparing a plan", "Running file_reader", "Checking the answer"): show it as an indicator |
| `token` | a piece of the answer: append it to the reply |
| `reset` | the text shown so far is not the answer: clear the reply |
| `final` | the whole answer, once: replace the reply with it |
| `error` | the request failed; the content explains why |
| `complete` | always last |

`final.metadata` holds `outcome`, `iteration` (steps used) and `max_iteration`:

| `outcome` | Meaning |
|---|---|
| `answered` | a normal answer |
| `awaiting_approval` | the reply is a plan: answer **yes** to run it, or say what to change |
| `clarification` | the reply is a question: the agent needs more detail |
| `best_effort` | the answer did not pass the agent's own check within the retry limit |
| `budget_exhausted` | the agent reached its step limit before finishing |

### A conversation

1. Ask for something that needs data: "count the EQD positions". The agent replies with a
   numbered plan (`awaiting_approval`). Nothing has run yet.
2. Reply **yes** to run it, or ask for a change ("also include FX") to get a new plan. Any
   other message drops the plan.
3. The agent runs its tools, checks the result, and answers (`answered`).

Questions that need no tools ("what is VaR?") are answered straight away. Follow-ups
("explain more", "only for last week") are understood from the conversation.

---

## Agent engines

`AGENT_ENGINE` picks the engine. Both serve the same API, follow the same routes (plan
first, tools only after approval, a check of the answer) and read the same configuration.

| | `langgraph` (default) | `anthropic_sdk` |
|---|---|---|
| Streaming | the accepted answer, in one piece | live, token by token; `reset` clears a draft that gets replaced |
| Storage | MongoDB, SQLite fallback | MongoDB only |
| Load | light | one Claude Code process per live conversation: keep `MAX_CONCURRENT_STREAMS` low |
| Built-in file tools | — | optional, see `AGENT_SDK_TOOLS` |

The `anthropic_sdk` engine speaks the Anthropic Messages API. Because the models are served
with the OpenAI convention, the agent starts a LiteLLM gateway that translates between the
two, and stops it on shutdown. Locally it runs through `uvx`; the Docker image ships it. To
do without it once the model server speaks Anthropic Messages: `LITELLM_ENABLED=false` and
`ANTHROPIC_BASE_URL`.

---

## Environment variables

`.env.example` lists them all, grouped. Variables marked **required** are read by the
configuration files and must be present, even when the engine you run does not use them
(an empty value is fine there).

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `debug` | which `config/<APP_ENV>/` folder is read |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. Each line shows the request id (`-` outside a request) |
| `CONFIGURATION_DIR` | `./config` | where `config/` is |
| `DEPLOYMENT_ENVIRONMENT` | `unknown` | label put on telemetry |
| `AGENT_ENGINE` | `langgraph` | `langgraph` or `anthropic_sdk` |
| `MAX_CONCURRENT_STREAMS` | `200` | open streams accepted at once; above it the API answers 503 |
| `LLM_BASE_URL` | **required** | the OpenAI-compatible model server, e.g. `http://sirius:8090/v1` |
| `TOOLBOX_URL` | **required** | the MCP server, e.g. `http://127.0.0.1:8001/mcp` |
| `DB_MONGO_CHECKPOINT_HOST` / `_PORT` / `_NAME` / `_USERNAME` / `_PASSWORD` | **required** | conversation storage in MongoDB |
| `DB_SQLITE_CHECKPOINT_HOST` / `_NAME` | **required** | SQLite fallback (folder + file name), used by `langgraph` only |
| `OTEL_HOST` / `OTEL_PORT` | **required** | OpenTelemetry collector (gRPC, usually port `4317`); empty host = telemetry off |

`anthropic_sdk` engine only:

| Variable | Default | Meaning |
|---|---|---|
| `WORKING_DIRECTORY` | `./working_directory` | the agent's working directory: the only place its built-in file tools can reach. Use the same path on every instance. Mount the toolbox's `WORKING_DIRECTORY` volume at the same path to share files |
| `AGENT_SDK_TOOLS` | empty | built-in file tools, usable only after a plan is approved: any of `Read`, `Write`, `Edit`, `Glob`, `Grep` (`Edit` requires `Read`) |
| `LITELLM_ENABLED` | `true` | start the LiteLLM gateway |
| `LITELLM_COMMAND` | `uvx --from litellm[proxy]==1.102.1 litellm` | how to launch LiteLLM |
| `LITELLM_HOST` / `LITELLM_PORT` | `127.0.0.1` / `4000` | where it listens |
| `ANTHROPIC_BASE_URL` | — | required when `LITELLM_ENABLED=false`: an Anthropic Messages endpoint |
| `ANTHROPIC_AUTH_TOKEN` | `none` | token sent to that endpoint |

---

## Configuration files

Everything lives under `config/`: `config/root.yml` lists what exists, and
`config/<APP_ENV>/connector/*.yml` and `config/<APP_ENV>/operation/*.yml` define it.

| File | What you set there |
|---|---|
| `connector/api.yml` | the model server (URL from `LLM_BASE_URL`), timeout, retries |
| `connector/mcp.yml` | the MCP servers (URL from `TOOLBOX_URL`, auth) |
| `connector/database.yml` | MongoDB and SQLite storage, conversation TTL |
| `connector/telemetry.yml` | OpenTelemetry collector |
| `operation/llm.yml` | the models users can pick, and their parameters |
| `operation/agent.yml` | which model each step of the agent uses |

### Models (`operation/llm.yml`)

One entry per model. Parameters:

| Parameter | Meaning | Default |
|---|---|---|
| `temperature` | sampling temperature | `0.0` |
| `max_output_tokens` | longest reply | `8000` |
| `max_context_tokens` | the model's context window | `8000` |
| `max_iterations` | step budget per message (model calls while working with tools) | `10` |
| `max_reflection_retries` | how many times a rejected answer is rewritten | `2` |
| `use_streaming` | `langgraph`: send the accepted answer as a `token` event before `final` | `false` |
| `reasoning_effort` | `null`: the model answers without thinking. `low`, `medium` or `high`: it thinks first (Qwen models; the level makes no difference for them, and models without a thinking mode ignore it). Thinking makes answers much slower and uses output tokens: give such a model `max_output_tokens` of 8000 or more | `null` |

The models are the ones the homelab model server runs (`homelab-infra/llm/config.yml`):
`qwen3-8b` (always loaded), `qwen3.5-0.8b`, `qwen3-1.7b`, `qwen3.5-2b`, `lfm2-8b-a1b`,
`ministral-8b-instruct-2410`, `gigachat3.1-10b-a1.8b` and `qwen3-14b`. Keep each
`max_context_tokens` equal to the server's `--ctx-size` for that model (16K, or 32K for the
models under 3B), and `max_output_tokens` well below it.

**Adding a model:** add its entry to `operation/llm.yml`, list it under `operation:` in
`config/root.yml`, add it to `MODEL_ALIASES` in `src/bootstrap/di/agent_di.py` (the list of
models requests may name), and to the model list of your UI. An entry's key cannot contain a
dot: write `qwen3_5-2b` as the key and `model: qwen3.5-2b` (the name the server knows) in its
parameters; `MODEL_ALIASES` maps the name requests use to that key.

### Model per step (`operation/agent.yml`)

The model named in the request always writes the answer. The other steps each have an entry:

| Entry | Step | Default |
|---|---|---|
| `agent_context` | understands each message and picks the route | `qwen3-8b` |
| `agent_plan` | writes the plan | `null` |
| `agent_reflection` | checks the answer before it is sent | `null` |
| `agent_summary` | summarises long conversations (`langgraph` only) | `qwen3-8b` |

- `parameters.model: <model name>`: the step always uses that model, with the parameters
  written in the same entry.
- `parameters.model: null`: the step uses the model named in the request, with its
  `llm.yml` parameters.

Changing a step's model is a config edit and a restart.

### Using an external MCP server instead of the bundled toolbox

The agent doesn't care whether `TOOLBOX_URL` points at the `agent_toolbox` service or
someone else's MCP server. Set `TOOLBOX_URL` in `.env`, or edit `base_url` in
`connector/mcp.yml`.

### Using more than one MCP server at once

Add a second connector in `connector/mcp.yml`, with a name starting with `external_mcp_`:

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

Restart the agent. Every `external_mcp_*` connector is picked up at boot and its tools join
the same catalogue. The boot log confirms what was found:

```
Discovered 2 MCP tools from 'toolbox': users_tables, python_executor
Discovered 5 MCP tools from 'external_mcp_analytics': ...
```

---

## Logging

Standard Python `logging`, one format for every line (set up by pycraftcore's
`configure_logging`):

```
2026-09-27 10:47:47.785 | INFO     | demo-logging-123 | stream_agent_controller:execute:50 - stream request accepted
```

- `LOG_LEVEL` sets the level (`INFO` by default).
- The third column is the request id (the `X-Request-ID` header, or the conversation id for
  `/v1/stream`); it appears on every line of that request by itself, `-` outside one. Do not
  put it in log messages.
- With `OTEL_HOST`/`OTEL_PORT` set, the same lines also go to the OpenTelemetry collector
  (Loki), with `request_id` as an attribute.

---

## Development

```bash
make check       # lint + typecheck + tests
make test
make lint
make format
make typecheck
make evaluation  # live quality evaluation: needs a real model server and toolbox
```

### Pre-commit hooks

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pre-commit run --all-files --hook-stage pre-commit
```

---

## Deploying

```bash
make docker_build     # image agent:local
```

The image listens on port 8000, reads its configuration from `/app/config` and ships
LiteLLM for the `anthropic_sdk` engine. Set the environment variables above on the
container; for `anthropic_sdk`, mount a volume at `WORKING_DIRECTORY`
(`/data/working_directory` in the image).
