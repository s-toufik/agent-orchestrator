import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request

from agent_orchestrator.adapter.inbound.web.controller.stream_agent_controller import (
    StreamAgentController,
)
from agent_orchestrator.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema
from agent_orchestrator.adapter.outbound.event.queue_turn_event_stream import QueueTurnEventStream
from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from agent_orchestrator.domain.workflow.step import Step

REQUEST = AgentRequestSchema(message="hi", model_name="m", request_id="r1")


class ScriptedUseCase:
    def __init__(self, events: list[TurnEvent]) -> None:
        self._events = events

    async def handle(self, request, events) -> None:
        for event in self._events:
            await events.publish(event)
        await events.complete()


def _finished(text: str) -> TurnEvent:
    turn = Turn(request="hi", model="m", settings=TurnSettings(max_steps=6))
    turn.finish(Answer.answered(text))
    return TurnEvent.finished(turn)


async def _drain(response) -> list[str]:
    return [chunk.decode("utf-8") async for chunk in response.body_iterator]


async def test_a_turn_streams_its_status_then_the_answer_then_complete(logger) -> None:
    turn = Turn(request="hi", model="m")
    events = [
        TurnEvent.entering(Step.UNDERSTAND, turn),
        TurnEvent.answer_delta("Hel"),
        _finished("Hello"),
    ]
    controller = StreamAgentController(ScriptedUseCase(events), QueueTurnEventStream, logger)

    response = await controller.execute(REQUEST)
    chunks = await _drain(response)

    assert "event: status" in chunks[0] and "Understanding your request" in chunks[0]
    assert "event: token" in chunks[1] and "Hel" in chunks[1]
    assert "event: final" in chunks[2]
    final_payload = json.loads(chunks[2].split("\n")[1].removeprefix("data: "))
    assert final_payload["content"] == "Hello"
    assert final_payload["session_id"] == "r1"
    assert final_payload["metadata"] == {
        "iteration": "0",
        "max_iteration": "6",
        "outcome": "answered",
    }
    assert "event: complete" in chunks[3]


async def test_admission_is_released_after_a_stream_completes(logger) -> None:
    events: list[TurnEvent] = []
    controller = StreamAgentController(
        ScriptedUseCase(events), QueueTurnEventStream, logger, max_concurrent_streams=1
    )

    response = await controller.execute(REQUEST)
    await _drain(response)

    response2 = await controller.execute(REQUEST)
    await _drain(response2)


async def test_admission_is_released_when_setup_itself_raises(logger) -> None:
    def broken_stream_events():
        raise RuntimeError("queue construction failed")

    controller = StreamAgentController(
        ScriptedUseCase([]), broken_stream_events, logger, max_concurrent_streams=1
    )

    with pytest.raises(RuntimeError, match="queue construction failed"):
        await controller.execute(REQUEST)

    await asyncio.wait_for(controller._admission.acquire(), timeout=1)


async def test_rejects_with_503_when_at_capacity(logger) -> None:
    controller = StreamAgentController(
        ScriptedUseCase([]), QueueTurnEventStream, logger, max_concurrent_streams=1
    )
    await controller._admission.acquire()

    with pytest.raises(HTTPException) as excinfo:
        await controller.execute(REQUEST)

    assert excinfo.value.status_code == 503
    assert logger.messages("warning")


async def test_client_disconnect_stops_the_stream_and_cancels_the_use_case(logger) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingUseCase:
        async def handle(self, request, events) -> None:
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    controller = StreamAgentController(HangingUseCase(), QueueTurnEventStream, logger)

    events = QueueTurnEventStream()
    generator = controller._event_generator(
        request=REQUEST,
        events=events,
        use_case_task=asyncio.create_task(HangingUseCase().handle(REQUEST, events)),
        starlette_request=cast(Request, DisconnectedRequest()),
    )

    with pytest.raises(asyncio.CancelledError):
        async for _ in generator:
            pass

    assert logger.messages("warning")


async def test_an_unexpected_error_yields_an_error_frame_then_reraises(logger) -> None:
    class BrokenEvents:
        async def publish(self, event: TurnEvent) -> None: ...
        async def complete(self) -> None: ...

        async def __aiter__(self) -> AsyncIterator[TurnEvent]:
            raise RuntimeError("queue broke")
            yield  # pragma: no cover

    controller = StreamAgentController(ScriptedUseCase([]), QueueTurnEventStream, logger)
    done_task = asyncio.create_task(asyncio.sleep(0))
    await done_task

    generator = controller._event_generator(
        request=REQUEST,
        events=BrokenEvents(),
        use_case_task=done_task,
        starlette_request=None,
    )

    chunks = []
    with pytest.raises(RuntimeError, match="queue broke"):
        async for chunk in generator:
            chunks.append(chunk.decode("utf-8"))

    assert len(chunks) == 1
    assert "event: error" in chunks[0]
    payload = json.loads(chunks[0].split("\n")[1].removeprefix("data: "))
    assert payload["type"] == "error"
    assert "queue broke" in payload["content"]
    assert logger.messages("error")
