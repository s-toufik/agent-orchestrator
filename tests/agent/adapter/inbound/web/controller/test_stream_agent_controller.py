import asyncio
import json
from typing import Any, cast

import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request

from agent.adapter.inbound.web.controller.stream_agent_controller import StreamAgentController
from agent.adapter.inbound.web.schema.agent_request_schema import AgentRequestSchema
from agent.domain.enum.agent_message_status import MessageStreamType
from agent.domain.model.agent_message_stream import AgentMessageStream

REQUEST = AgentRequestSchema(message="hi", model_name="m", request_id="r1")


class FakeSSEQueue:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[AgentMessageStream] = asyncio.Queue()

    async def token(self, value: str) -> None: ...
    async def final(self, value: str, metadata: dict | None = None) -> None: ...
    async def error(self, value: str) -> None: ...
    async def complete(self) -> None: ...


class ScriptedUseCase:
    def __init__(self, events: list[AgentMessageStream]) -> None:
        self._events = events

    async def execute(self, request, events) -> None:
        for event in self._events:
            await events.queue.put(event)


async def _drain(response) -> list[str]:
    return [chunk.decode("utf-8") async for chunk in response.body_iterator]


async def test_a_normal_stream_yields_token_then_final_then_complete(logger) -> None:
    events = [
        AgentMessageStream(type=MessageStreamType.TOKEN, content="Hel"),
        AgentMessageStream(type=MessageStreamType.TOKEN, content="lo"),
        AgentMessageStream(
            type=MessageStreamType.FINAL, content="Hello", metadata={"iteration": "1"}
        ),
        AgentMessageStream(type=MessageStreamType.COMPLETE, content=""),
    ]
    controller = StreamAgentController(ScriptedUseCase(events), FakeSSEQueue, logger)

    response = await controller.execute(REQUEST)
    chunks = await _drain(response)

    assert "event: token" in chunks[0]
    assert "event: final" in chunks[2]
    final_payload = json.loads(chunks[2].split("\n")[1].removeprefix("data: "))
    assert final_payload["content"] == "Hello"
    assert final_payload["metadata"] == {"iteration": "1"}
    assert "event: complete" in chunks[3]


async def test_admission_is_released_after_a_stream_completes(logger) -> None:
    events = [AgentMessageStream(type=MessageStreamType.COMPLETE, content="")]
    controller = StreamAgentController(
        ScriptedUseCase(events), FakeSSEQueue, logger, max_concurrent_streams=1
    )

    response = await controller.execute(REQUEST)
    await _drain(response)

    # A second stream must be able to acquire the single permit again.
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

    # The permit must have been released, or a second call would hang forever
    # waiting on the semaphore -- prove it by acquiring it with a timeout.
    await asyncio.wait_for(controller._admission.acquire(), timeout=1)


async def test_rejects_with_503_when_at_capacity(logger) -> None:
    controller = StreamAgentController(
        ScriptedUseCase([]), FakeSSEQueue, logger, max_concurrent_streams=1
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
        async def execute(self, request, events) -> None:
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    controller = StreamAgentController(HangingUseCase(), FakeSSEQueue, logger)

    events = FakeSSEQueue()
    generator = controller._event_generator(
        request_id="r1",
        request=REQUEST,
        events=events,
        use_case_task=asyncio.create_task(HangingUseCase().execute(REQUEST, events)),
        starlette_request=cast(Request, DisconnectedRequest()),
    )

    with pytest.raises(asyncio.CancelledError):
        async for _ in generator:
            pass

    assert logger.messages("warning")


async def test_an_unexpected_error_yields_an_error_frame_then_reraises(logger) -> None:
    class RaisingQueue:
        async def get(self):
            raise RuntimeError("queue broke")

    class BrokenEvents:
        queue: Any = RaisingQueue()

        async def token(self, value: str) -> None: ...
        async def final(self, value: str, metadata: dict | None = None) -> None: ...
        async def error(self, value: str) -> None: ...
        async def complete(self) -> None: ...

    controller = StreamAgentController(ScriptedUseCase([]), FakeSSEQueue, logger)
    done_task = asyncio.create_task(asyncio.sleep(0))
    await done_task

    generator = controller._event_generator(
        request_id="r1",
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
    assert "event: final" in chunks[0]
    payload = json.loads(chunks[0].split("\n")[1].removeprefix("data: "))
    assert "queue broke" in payload["error"]
    assert logger.messages("error")
