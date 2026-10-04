import asyncio
from collections.abc import AsyncIterator

import pytest

from agent_orchestrator.application.port.inbound.agent_request import AgentRequest
from agent_orchestrator.application.use_case.handle_message import (
    AGENT_UNAVAILABLE_MESSAGE,
    HandleMessage,
)
from agent_orchestrator.domain.event.turn_event import TurnEvent, TurnEventKind
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from tests.agent_orchestrator.application.fakes import FakeModels, ListEventStream

REQUEST = AgentRequest(message="hello", model_name="m", request_id="c1")


class RecordingRunner:
    def __init__(self, failure: Exception | None = None) -> None:
        self._failure = failure
        self.turns: list[tuple[str, Turn]] = []

    async def run(self, conversation_id: str, turn: Turn) -> AsyncIterator[TurnEvent]:
        self.turns.append((conversation_id, turn))
        if self._failure is not None:
            raise self._failure
        yield TurnEvent.finished(turn)


async def test_the_turn_is_run_on_the_conversation_and_its_events_published(logger) -> None:
    runner, events = RecordingRunner(), ListEventStream()
    settings = TurnSettings(max_steps=3)

    await HandleMessage(runner, FakeModels(settings), logger).handle(REQUEST, events)

    conversation_id, turn = runner.turns[0]
    assert conversation_id == "c1"
    assert (turn.request, turn.model, turn.settings) == ("hello", "m", settings)
    assert [event.kind for event in events.events] == [TurnEventKind.FINISHED]
    assert events.completed


async def test_an_unavailable_agent_reports_a_friendly_error(logger) -> None:
    events = ListEventStream()
    runner = RecordingRunner(AgentUnavailableException("llm down"))

    await HandleMessage(runner, FakeModels(), logger).handle(REQUEST, events)

    assert events.events == [TurnEvent.failed(AGENT_UNAVAILABLE_MESSAGE)]
    assert events.completed


async def test_any_other_failure_reports_the_trace(logger) -> None:
    events = ListEventStream()

    await HandleMessage(RecordingRunner(RuntimeError("boom")), FakeModels(), logger).handle(
        REQUEST, events
    )

    assert events.events[0].kind is TurnEventKind.FAILED
    assert "boom" in events.events[0].error
    assert logger.messages("error")


async def test_an_unknown_model_is_reported_without_running_a_turn(logger) -> None:
    runner, events = RecordingRunner(), ListEventStream()
    request = AgentRequest(message="hello", model_name="unknown", request_id="c1")

    await HandleMessage(runner, FakeModels(), logger).handle(request, events)

    assert runner.turns == []
    assert events.events[0].kind is TurnEventKind.FAILED
    assert events.completed


async def test_the_auto_approve_choice_reaches_the_turn(logger) -> None:
    runner = RecordingRunner()
    request = AgentRequest(message="go", model_name="m", request_id="c1", auto_approve=True)

    await HandleMessage(runner, FakeModels(), logger).handle(request, ListEventStream())

    assert runner.turns[0][1].options.auto_approve is True


async def test_a_finished_turn_is_logged_with_its_outcome_and_duration(logger) -> None:
    await HandleMessage(RecordingRunner(), FakeModels(), logger).handle(REQUEST, ListEventStream())

    [finished] = logger.messages("info")
    assert finished.startswith("turn finished: no answer, 0/")
    assert finished.endswith("s")


class _EndlessRunner:
    async def run(self, conversation_id: str, turn: Turn) -> AsyncIterator[TurnEvent]:
        await asyncio.Event().wait()
        yield TurnEvent.finished(turn)


async def test_a_cancelled_turn_is_logged_and_still_cancelled(logger) -> None:
    events = ListEventStream()
    task = asyncio.create_task(
        HandleMessage(_EndlessRunner(), FakeModels(), logger).handle(REQUEST, events)
    )
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert logger.messages("warning")[0].startswith("turn cancelled after ")
    assert events.completed
