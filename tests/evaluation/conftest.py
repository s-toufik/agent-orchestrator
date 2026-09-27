import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.langgraph_agent import LangGraphAgent
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import unpack_state
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest
from bootstrap.configuration.settings import ProcessSettings
from bootstrap.di.langgraph_di import LangGraphDI
from tests.evaluation.local_judge_model import LocalJudgeModel

pytestmark = pytest.mark.evaluation

REAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
EVAL_MODEL: str = os.getenv("EVAL_MODEL", "gpt_oss_20b")

RunAgent = Callable[[str], Awaitable[tuple[AgentMessageStream, AgentState]]]


def _settings(role: str) -> ProcessSettings:
    return ProcessSettings(
        role=role,
        environment="debug",
        configuration_directory=REAL_CONFIG_DIR,
    )


@pytest.fixture
def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_HOST", str(tmp_path))
    monkeypatch.setenv("DB_SQLITE_CHECKPOINT_NAME", "checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_HOST", "localhost")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PORT", "27017")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_NAME", "checkpoint")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_USERNAME", "test")
    monkeypatch.setenv("DB_MONGO_CHECKPOINT_PASSWORD", "test")


@pytest.fixture
async def agent_di(_base_env) -> AsyncIterator[LangGraphDI]:
    di = LangGraphDI(_settings("agent-orchestrator"))
    try:
        yield di
    finally:
        await di._close_mcp_session_factories()
        await di._close_llm_http_client()
        await di._shutdown_telemetry()


@pytest.fixture
def judge_model(agent_di: LangGraphDI) -> LocalJudgeModel:
    return LocalJudgeModel(agent_di._llm_for_model(EVAL_MODEL, use_streaming=False))


@pytest.fixture
async def run_agent(agent_di: LangGraphDI) -> RunAgent:
    checkpointer = await agent_di._checkpointer()
    tool_registry = await agent_di._tool_registry()
    graph = agent_di._build_graph(EVAL_MODEL, checkpointer, tool_registry)
    agent = LangGraphAgent({EVAL_MODEL: graph})

    async def _final(message: str, request_id: str) -> AgentMessageStream:
        request = AgentRequest(message=message, model_name=EVAL_MODEL, request_id=request_id)
        events = [event async for event in agent.stream(request)]
        return events[-1]

    async def _run(question: str) -> tuple[AgentMessageStream, AgentState]:
        request_id = str(uuid.uuid4())
        message = await _final(question, request_id)
        if message.metadata.get("outcome") == TurnOutcome.AWAITING_APPROVAL:
            message = await _final("yes", request_id)
        snapshot = await graph.aget_state({"configurable": {"thread_id": request_id}})
        return message, unpack_state(snapshot.values)

    return _run
