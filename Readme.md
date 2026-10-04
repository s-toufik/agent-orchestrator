# agent-orchestrator

An agent served over HTTP. Send it a message: it works out what you want, answers straight away when it can, proposes a plan when the request needs tools, runs the tools once you approve, checks its own answer, and streams the reply back as server-sent events.

**[Using the agent](#using-the-agent)** — run it, talk to it, configure it.
**[Working on the agent](#working-on-the-agent)** — how it is built and how to change it.

```text
                ┌──────────────────────────────────────────┐
  your app ────►│  agent-orchestrator                      │
  (homelab-ui)  │                                          │
       ▲        │  understand ─► plan ─► act ─► review     │──── OpenAI API ───► model server
       │        │                         │                │
       │        │                         └─► run tools ───│──── MCP ──────────► agent-toolbox
       │        │                                          │                    (or any MCP server)
       └────────│  SSE: status · token · final · complete  │──── storage ──────► MongoDB / SQLite
                └──────────────────────────────────────────┘
```

---

## Using the agent

### Run it

Requirements: Python 3.14 and [uv](https://docs.astral.sh/uv/), an OpenAI-compatible model server, and optionally an MCP server for tools (the agent starts without one) and MongoDB for conversations (a local SQLite file is used when MongoDB is unreachable).

```bash
uv sync --group dev
cp .env.example .env      # fill in the model server, MCP server and storage
make run
```

| | Endpoint |
|---|---|
| Chat (SSE) | `POST http://<host>:<port>/v1/stream` |
| Models | `GET http://<host>:<port>/v1/models` |
| Health | `GET http://<host>:<port>/actuator/health` (also `/health/liveness`, `/health/readiness`) |
| Version | `GET http://<host>:<port>/actuator/info` |

`/health/readiness` answers `503` until the agent has booted, then `200`. The boot log lists the tools found on each MCP server; a server that is down is skipped with a warning.

With Docker: `make docker_build` builds the image; it reads its configuration from `/app/config` and takes the environment variables below.

### List the models

`GET /v1/models` returns the models a request may name, and the steps that always use one model whatever the request asks:

```json
{
  "models": [{ "name": "qwen3-8b", "context_tokens": 16384, "max_output_tokens": 4096, "thinking": false }],
  "pinned_steps": { "act": "<model>", "understand": "<model>" }
}
```

A client builds its model picker from `models` rather than keeping its own list. When `pinned_steps` covers every step, the choice has no effect on which model answers.

### Send a message

```bash
curl -N http://<host>:<port>/v1/stream \
  -H 'Content-Type: application/json' \
  -d '{"message": "how many users signed up today?", "model_name": "qwen3-8b", "request_id": "my-conversation"}'
```

| Field | Meaning |
|---|---|
| `message` | What you say (non-empty) |
| `model_name` | One of the models in `config/<env>/operation/llm.yml` |
| `request_id` | The conversation id. Same value for every message of one conversation; a new value starts an empty one |
| `auto_approve` | Optional, default `false`. `true` runs a plan as soon as it is written instead of waiting for your "yes" |

### Read the reply

A server-sent event stream; each frame is `event: <type>` + `data: <json>`. New event types and outcomes may be added: a client should ignore an event type it doesn't know, and treat an unknown `outcome` like `answered`.

| Event | What to do with it |
|---|---|
| `status` | Progress ("Understanding your request", "Preparing a plan", "Running file_reader", "Checking the answer"): show it as an indicator |
| `token` | A piece of the answer as it is written, sent for answers that need no review when `AGENT_ANSWER_DELIVERY=stream`: append the pieces, then let `final` replace them |
| `final` | The complete answer: show it (it replaces any `token` pieces) |
| `error` | The request failed; the content says why |
| `complete` | Always last |

`final.metadata` holds `iteration` (steps used), `max_iteration` and `outcome`:

| `outcome` | Meaning |
|---|---|
| `answered` | A normal answer |
| `awaiting_approval` | The reply is a plan: answer **yes** to run it, or say what to change |
| `clarification` | The reply is a question: the agent needs more detail |
| `best_effort` | The answer did not pass the agent's own check within the retry limit |
| `budget_exhausted` | The agent reached its step limit before finishing |

### How a message is handled

Every message goes through **understand** first. What the agent understood decides the route; after acting, what the model produced decides the next step. The routes below are today's rules: they all live in one table, `TurnPolicy`, so new steps and rules are added there (see [The routing table](#the-routing-table)) and this section follows.

```text
                         ┌─ task, revision ──► PLAN ─┬─ auto-approve ──► ACT
                         │                           └─ otherwise ─────► FINISH: waits for "yes"
  message ──► UNDERSTAND ┼─ ambiguous ───────► CLARIFY ──► FINISH: a question
                         └─ direct, follow-up,
                            "yes" to a plan ─► ACT

  ACT ─┬─ needs tools, has no plan ───► PLAN
       ├─ wants tools, no steps left ─► FINISH: step limit reached
       ├─ calls tools ────────────────► RUN TOOLS ──► ACT
       ├─ plain direct answer ────────► FINISH: no check needed
       └─ otherwise ──────────────────► REVIEW ─┬─ rejected, can retry ─► FEEDBACK ─► ACT
                                                └─ accepted, or no retry ─► FINISH

  FINISH ──► SUMMARIZE: folds old messages away when the history is long ──► reply sent
```

Some typical journeys:

| You send | Route | Reply |
|---|---|---|
| "What can you do?" | understand → act → finish | `answered`, straight away |
| "How many users signed up today?" | understand → plan → finish | `awaiting_approval`: the plan |
| "yes" | understand → act → run tools → act → review → finish | `answered`, from the tool results |
| The same task with `auto_approve: true` | understand → plan → act → run tools → act → review → finish | `answered`, in one go |
| "Explain report.md" read as a plain question | understand → act → plan → … | The agent notices it needs a tool and plans after all |
| "and the other one?" with nothing to refer to | understand → clarify → finish | `clarification`: a question back |
| "shorter" after an answer | understand → act → review → finish | `answered`, reworked from the conversation |

A pending plan survives while you discuss it (a revision or an unclear message) and is dropped as soon as you move on to something else.

### Configure it

**Environment variables** — `.env.example` is the reference; the main ones:

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `debug` | Which `config/<APP_ENV>/` folder is read |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `CONFIGURATION_DIR` | `./config` | Where `config/` is |
| `DEPLOYMENT_ENVIRONMENT` | `unknown` | Label shown by `/actuator/info` and on telemetry |
| `MAX_CONCURRENT_STREAMS` | `200` | Open streams accepted at once; above it the API answers `503` |
| `AGENT_ANSWER_DELIVERY` | `stream` | `stream`: answers that need no review are sent piece by piece as `token` events, then `final`. `whole`: only `final` |
| `LLM_BASE_URL`, `LLM_API_KEY` | required | The OpenAI-compatible model server, e.g. `http://<llm-server>:<port>/v1`, and its key (empty for a local server) |
| `TOOLBOX_URL` | required | The MCP server, e.g. `http://<toolbox-host>:<port>/mcp` |
| `DB_MONGO_CHECKPOINT_HOST` / `_PORT` / `_NAME` / `_USERNAME` / `_PASSWORD` | required | Conversation storage in MongoDB |
| `DB_SQLITE_CHECKPOINT_HOST` / `_NAME` | required | SQLite fallback (folder and file name) |
| `OTEL_HOST`, `OTEL_PORT` | required | OpenTelemetry collector (gRPC); an empty host turns telemetry off |

"Required" variables must exist; an empty value is fine when the feature is unused.

**Configuration files** live in `config/`: `root.yml` lists what exists, `config/<APP_ENV>/connector/*.yml` and `operation/*.yml` define it.

| File | What you set there |
|---|---|
| `connector/api.yml` | The model server, timeout, retries |
| `connector/mcp.yml` | The MCP servers |
| `connector/database.yml` | MongoDB and SQLite, conversation TTL |
| `connector/telemetry.yml` | The OpenTelemetry collector |
| `operation/llm.yml` | The models users can pick, and their parameters |
| `operation/agent.yml` | Which model each step uses |

**Models** (`operation/llm.yml`), one entry per model:

| Parameter | Meaning | Default |
|---|---|---|
| `model` | The name the model server knows, and the `model_name` requests use | the entry's `name` |
| `temperature` | Sampling temperature | `0.0` |
| `max_output_tokens` | Longest reply | `8000` |
| `max_context_tokens` | The model's context window | `8000` |
| `max_iterations` | Step budget per message | `10` |
| `max_reflection_retries` | How many times a rejected answer is rewritten | `2` |
| `reasoning_effort` | `null`: no thinking. `low` / `medium` / `high`: think first (slower, needs a larger `max_output_tokens`) | `null` |

To add a model: add its entry, list it under `operation:` in `config/root.yml`, and restart; it then appears in `GET /v1/models`. An entry's key cannot contain a dot: write `qwen3_5-2b` as the key and `model: qwen3.5-2b` in its parameters.

**Model per step** (`operation/agent.yml`): every step that calls a model has an entry named after its role — today `agent_context` (understand), `agent_plan`, `agent_act` (answer and tools), `agent_reflection` (review) and `agent_summary`. A step added later that calls a model gets its own entry the same way.

- `parameters.model: null` — the step uses the model named in the request.
- `parameters.model: <name>` — the step always uses that model, with the parameters written in the same entry. When `agent_act` is fixed, its entry also sets the turn's budget (`max_iterations`, `max_reflection_retries`, `max_context_tokens`).

**More MCP servers:** add a connector whose name starts with `external_mcp_` in `connector/mcp.yml`, list it under `mcp:` in `config/root.yml`, and restart. Its tools join the same catalogue:

```yaml
connector:
  external_mcp_analytics:
    name: external_mcp_analytics
    type: mcp
    base_url: https://<analytics-host>/mcp
    timeout: 30
    transport: streamable_http
    auth:
      type: token
      key_name: Authorization
      key_value: ${oc.env:ANALYTICS_MCP_TOKEN,''}
```

### Logs

One format for every line; the third column is the request id (the conversation id on `/v1/stream`), `-` outside a request:

```text
2026-09-27 10:47:47.785 | INFO     | my-conversation | stream_agent_controller:execute:45 - stream request accepted
2026-09-27 10:47:52.310 | INFO     | my-conversation | handle_message:handle:42 - turn finished: answered, 1/20 steps in 4.5s
```

Every turn ends with a `turn finished` line (outcome, steps used, duration) or a `turn cancelled after …` warning when the client left first.

With `OTEL_HOST` set, the same lines also go to the OpenTelemetry collector, with `request_id` as an attribute.

---

## Working on the agent

### Architecture

Hexagonal: the rules in the middle know nothing about HTTP, LangChain, LangGraph or MCP. `tests/architecture/test_boundaries.py` fails the build if an import crosses a boundary.

```text
  ┌─ bootstrap ──────────────────────────────────────────────────────────────┐
  │  configuration, dependency wiring (AgentDI), FastAPI app, routers        │
  │  ┌─ adapter ──────────────────────────────────────────────────────────┐  │
  │  │  inbound:  HTTP controller, SSE presenter                          │  │
  │  │  outbound: LangChain models, MCP tools, LangGraph runtime, events  │  │
  │  │  ┌─ application ────────────────────────────────────────────────┐  │  │
  │  │  │  HandleMessage use case, one handler per step, ports         │  │  │
  │  │  │  ┌─ domain ───────────────────────────────────────────────┐  │  │  │
  │  │  │  │  Conversation, Turn, Plan, Answer, TurnPolicy          │  │  │  │
  │  │  │  │  plain dataclasses, no framework                       │  │  │  │
  │  │  │  └────────────────────────────────────────────────────────┘  │  │  │
  │  │  └──────────────────────────────────────────────────────────────┘  │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────────┘
```

### Project layout

| Path | What it holds |
|---|---|
| `src/agent_orchestrator/domain/` | `Conversation` (history, summary, pending plan), `Turn` (one message and its work), value objects, `TurnPolicy` (the routing table) |
| `src/agent_orchestrator/application/` | `HandleMessage`, the step handlers (`*_step.py`), and the ports they call (`Actor`, `Planner`, `ToolExecutor`, `WorkflowRunner`…) |
| `src/agent_orchestrator/adapter/inbound/web/` | `StreamAgentController`, `SsePresenter`, request and response schemas |
| `src/agent_orchestrator/adapter/outbound/llm/` | One LangChain adapter per model role, prompts, `ModelCatalog` |
| `src/agent_orchestrator/adapter/outbound/tool/` | MCP discovery and calls (`Toolbox`) |
| `src/agent_orchestrator/adapter/outbound/langgraph/` | The graph: one node per step, edges built from `TurnPolicy`, state codec, checkpointer |
| `src/bootstrap/` | Settings, `AgentDI` (wiring only), FastAPI application, routers |
| `config/` | YAML configuration |
| `tests/` | Unit, scenario, architecture and evaluation tests |

### A turn, end to end

1. `StreamAgentController` validates the body, starts `HandleMessage` in the background and returns the SSE response at once.
2. `HandleMessage` builds a `Turn` (the message, the model's budget, the user's options) and asks the `WorkflowRunner` to run it on the conversation `request_id`.
3. `LangGraphWorkflowRunner` loads the stored `Conversation` and runs the graph. Each node decodes the state, runs its step handler, and saves the state: LangGraph's checkpointer keeps the conversation between messages.
4. After each step, `TurnPolicy.next(step, turn)` picks the next one (the table below). Each node emits a `TurnEvent`, which `SsePresenter` turns into SSE frames.
5. `finish` settles the answer and appends the exchange to the history; `summarize` compacts it when it grows past half the context.

### The routing table

`domain/workflow/turn_policy.py` is an ordered list of transitions; after a step, the first row whose condition holds wins. The LangGraph edges are generated from it, and `test_agent_graph.py` checks they match.

| After | Condition | Next |
|---|---|---|
| understand | `needs_plan` (task, plan revision, or no intent) | plan |
| understand | `is_ambiguous` | clarify |
| understand | always | act |
| plan | `has_approved_plan` (auto-approved) | act |
| plan | always | finish |
| clarify | always | finish |
| act | `asks_for_plan_without_one` | plan |
| act | `asks_for_tools_out_of_steps` | finish |
| act | `asks_for_tools` | run_tools |
| act | `is_plain_direct_answer` | finish |
| act | always | review |
| run_tools, feedback | always | act |
| review | `retry_allowed` | feedback |
| review | always | finish |
| finish | always | summarize |
| summarize | always | end |

### Making changes

| To change | Edit |
|---|---|
| When a step follows another | `TurnPolicy.TRANSITIONS` (and a condition in `conditions.py`); the graph follows |
| What a step does | Its handler in `application/step/` |
| What a model is told | `adapter/outbound/llm/langchain/prompts/` |
| What the client sees for a step | `adapter/inbound/web/sse_presenter.py` |
| A LangGraph-only feature for one step (interrupt, subgraph) | Override `work` in that step's node in `adapter/outbound/langgraph/node/` |
| A new request option | `AgentRequestSchema` → `AgentRequest` → `TurnOptions`, read in the domain |
| A new step | A `Step` value, a handler in `application/step/`, a node class in `langgraph/node/` listed in `NODE_TYPES`, the handler in `AgentDI._steps`, and its rows in `TurnPolicy`; the graph's edges follow |
| A new model role | An `AgentRole` value, its entry in `operation/agent.yml` (and `root.yml`), and a LangChain adapter behind the step's port |
| A new source of tools | A connector in `connector/mcp.yml` (see [More MCP servers](#configure-it)); a non-MCP source is a new adapter behind the `ToolCatalog` and `ToolExecutor` ports |

### Tests and checks

```bash
make check        # ruff + ty + pytest
make test
make lint / make format / make typecheck
make evaluation   # live quality evaluation: needs a real model server and toolbox
```

`tests/agent_orchestrator/scenarios/` runs whole turns on two runners — LangGraph and a plain in-memory loop over the same steps and policy — and requires the same answers, events and stored conversation from both.

Pre-commit hooks:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

### Deploying

`make docker_build` builds `agent:local`. The image listens on port 8000, reads its configuration from `/app/config`, and takes the environment variables listed above.
