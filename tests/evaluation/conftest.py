"""Fixtures for live agent-quality evaluation with DeepEval.

These tests drive the real LangGraph agent against a real toolbox -- real
tools, real sqlite -- and use the project's own local LLM (served via LM
Studio, see `connector.llm.base_url` in config) both as the agent-under-test
and as the DeepEval judge model. No external LLM provider (OpenAI,
Anthropic, ...) is ever called.

The agent and its toolbox are independently deployed services now, so
unlike the rest of this test suite these tests can't boot the toolbox
in-process: a real `agent_toolbox` (see the sibling repo) must already be
running and reachable at `TOOLBOX_URL` before this suite starts, exactly as
in normal operation.

They require a running local LLM server plus a running toolbox, and are
excluded from the default `pytest` run (see the `evaluation` marker and
`addopts` in pyproject.toml). Run them explicitly with:

    # in the agent_toolbox repo
    make run
    # in this repo
    pytest -m evaluation tests/evaluation
"""

import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from agent_orchestrator.adapter.outbound.langgraph.lang_agent import LangAgent
from agent_orchestrator.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.service.state_serialization import unpack_state
from agent_orchestrator.domain.model.agent_message import AgentMessage
from agent_orchestrator.domain.model.agent_request import AgentRequest
from bootstrap.configuration.settings import ProcessSettings
from bootstrap.di.agent_di import AgentDI
from tests.evaluation.local_judge_model import LocalJudgeModel

pytestmark = pytest.mark.evaluation

REAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
EVAL_MODEL: str = os.getenv("EVAL_MODEL", "gpt_oss_20b")

RunAgent = Callable[[str], Awaitable[tuple[AgentMessage, AgentState]]]


def _settings(role: str, port: int) -> ProcessSettings:
    return ProcessSettings(
        role=role,
        environment="debug",
        configuration_directory=REAL_CONFIG_DIR,
        host="0.0.0.0",
        port=port,
    )


@pytest.fixture
def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_DB_HOST", str(tmp_path))
    monkeypatch.setenv("CHECKPOINT_DB_NAME", "checkpoint")


@pytest.fixture
async def agent_di(_base_env) -> AsyncIterator[AgentDI]:
    di = AgentDI(_settings("agent-orchestrator", 8000))
    try:
        yield di
    finally:
        await di._close_mcp_session_factories()
        await di._close_llm_http_client()
        await di._shutdown_telemetry()


@pytest.fixture
def judge_model(agent_di: AgentDI) -> LocalJudgeModel:
    """The project's own local LLM, wrapped so DeepEval scores with it instead of OpenAI."""
    return LocalJudgeModel(agent_di._llm_for_model(EVAL_MODEL, use_streaming=False))


@pytest.fixture
async def run_agent(agent_di: AgentDI) -> RunAgent:
    """Runs one real turn through the real agent graph and returns
    (the reply, the full final state -- conversation, tool calls, tool outputs)."""
    checkpointer = await agent_di._checkpointer()
    tool_registry = await agent_di._tool_registry()
    graph = agent_di._build_graph(EVAL_MODEL, checkpointer, tool_registry)
    agent = LangAgent({EVAL_MODEL: graph})

    async def _run(question: str) -> tuple[AgentMessage, AgentState]:
        request_id = str(uuid.uuid4())
        message = await agent.run(
            AgentRequest(message=question, model_name=EVAL_MODEL, request_id=request_id)
        )
        snapshot = await graph.aget_state({"configurable": {"thread_id": request_id}})
        return message, unpack_state(snapshot.values)

    return _run
