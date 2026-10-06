import os
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from pathlib import Path

import pytest

from agent_orchestrator.adapter.outbound.llm.model_catalog import AgentRole
from agent_orchestrator.domain.exception.unknown_model_exception import UnknownModelException
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.turn import Turn
from bootstrap.configuration.settings import ProcessSettings
from bootstrap.di.agent_di import AgentDI
from tests.evaluation.harness.report import EvaluationReport
from tests.evaluation.harness.route_coverage import RouteCoverage
from tests.evaluation.harness.score import Score, threshold_from_env
from tests.evaluation.harness.session import EvalSession
from tests.evaluation.local_judge_model import LocalJudgeModel

EVALUATION_DIR = Path(__file__).resolve().parent
REAL_CONFIG_DIR = EVALUATION_DIR.parents[1] / "config"
REPORTS_DIR = EVALUATION_DIR / "reports"
EVAL_MODEL: str = os.getenv("EVAL_MODEL", "qwen3-14b")
COVERAGE = RouteCoverage()
REPORT = EvaluationReport(EVAL_MODEL, threshold_from_env(), COVERAGE)
WRITTEN: list[Path] = []

NewSession = Callable[[], EvalSession]
RunAgent = Callable[[str], Awaitable[tuple[Answer, Turn]]]
Scorecard = Callable[[Score], None]


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(EVALUATION_DIR):
            item.add_marker(pytest.mark.evaluation)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    # A score below the threshold fails with an AssertionError; anything else never got a score.
    if call.excinfo is not None and not call.excinfo.errisinstance(AssertionError):
        if item.path.is_relative_to(EVALUATION_DIR) and call.when in ("setup", "call"):
            REPORT.add_unscored(item.name, f"{call.excinfo.typename}: {call.excinfo.value}"[:300])
    return report


def pytest_sessionfinish(session: pytest.Session) -> None:
    if not REPORT.is_empty:
        WRITTEN.extend(REPORT.write(REPORTS_DIR, datetime.now()))


def pytest_terminal_summary(terminalreporter) -> None:
    if REPORT.is_empty:
        return
    terminalreporter.section("agent evaluation")
    for line in [*REPORT.summary(), *COVERAGE.report(), *(f"report: {path}" for path in WRITTEN)]:
        terminalreporter.write_line(line)


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
        await di.close()


@pytest.fixture
def judge_model(agent_di: AgentDI) -> LocalJudgeModel:
    return LocalJudgeModel(agent_di._chat_models.for_role(AgentRole.ACT, EVAL_MODEL))


@pytest.fixture
async def new_session(agent_di: AgentDI) -> NewSession:
    runner = await agent_di._workflow_runner(
        await agent_di._toolbox(), await agent_di._checkpointer()
    )
    catalog = agent_di._model_catalog
    try:
        settings = catalog.turn_settings(EVAL_MODEL)
    except UnknownModelException:
        names = "\n  ".join(model.name for model in catalog.listing().models)
        pytest.exit(f"EVAL_MODEL={EVAL_MODEL!r} is not a known model. Pick one of:\n  {names}")
    return lambda: EvalSession(runner, EVAL_MODEL, settings, COVERAGE)


@pytest.fixture
def run_agent(new_session: NewSession) -> RunAgent:
    async def _run(question: str) -> tuple[Answer, Turn]:
        exchange = await new_session().say(question, auto_approve=True)
        assert exchange.answer is not None
        return exchange.answer, exchange.turn

    return _run


@pytest.fixture
def threshold() -> float:
    return REPORT.threshold


@pytest.fixture
def scorecard() -> Scorecard:
    def _record(score: Score) -> None:
        REPORT.add(score)
        assert score.passed, score.explain()

    return _record
