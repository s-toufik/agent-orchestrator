import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from agent_orchestrator.adapter.outbound.langgraph.turn_state_codec import TurnStateCodec
from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.turn.turn import Turn
from bootstrap.configuration.settings import ProcessSettings
from bootstrap.di.agent_di import AgentDI
from tests.evaluation.local_judge_model import LocalJudgeModel

pytestmark = pytest.mark.evaluation

REAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
EVAL_MODEL: str = os.getenv("EVAL_MODEL", "qwen3-14b")

RunAgent = Callable[[str], Awaitable[tuple[Answer, Turn]]]


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
async def agent_di(_base_env) -> AsyncIterator[AgentDI]:
    di = AgentDI(_settings("agent-orchestrator"))
    try:
        yield di
    finally:
        await di._close_mcp_session_factories()
        await di._close_llm_http_client()
        await di._shutdown_telemetry()


@pytest.fixture
def judge_model(agent_di: AgentDI) -> LocalJudgeModel:
    return LocalJudgeModel(agent_di._chat_models.for_role(AgentRole.ACT, EVAL_MODEL))


@pytest.fixture
async def run_agent(agent_di: AgentDI) -> RunAgent:
    runner = await agent_di._workflow_runner(
        await agent_di._toolbox(), await agent_di._checkpointer()
    )
    settings = agent_di._model_catalog.turn_settings(EVAL_MODEL)
    codec = TurnStateCodec()

    async def _turn(message: str, conversation_id: str) -> Turn:
        async for _ in runner.run(conversation_id, Turn(message, EVAL_MODEL, settings)):
            pass
        snapshot = await runner.graph.aget_state({"configurable": {"thread_id": conversation_id}})
        return codec.decode_turn(snapshot.values)

    async def _run(question: str) -> tuple[Answer, Turn]:
        conversation_id = str(uuid.uuid4())
        turn = await _turn(question, conversation_id)
        if turn.answer is not None and turn.answer.outcome is Outcome.AWAITING_APPROVAL:
            turn = await _turn("yes", conversation_id)
        assert turn.answer is not None
        return turn.answer, turn

    return _run
