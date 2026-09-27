from collections.abc import AsyncIterator

from agent_orchestrator.adapter.outbound.streaming.sse_queue import SSEQueue
from agent_orchestrator.application.use_case.stream_agent_usecase import (
    AGENT_UNAVAILABLE_MESSAGE,
    StreamAgentUseCase,
)
from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest

REQUEST = AgentRequest(message="hello", model_name="qwen3-14b", request_id="r1")


class ScriptedAgent:
    def __init__(self, events: list[AgentMessageStream], error: Exception | None = None) -> None:
        self._events = events
        self._error = error

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentMessageStream]:
        for event in self._events:
            yield event
        if self._error:
            raise self._error


async def _run(agent: ScriptedAgent, logger) -> list[AgentMessageStream]:
    events = SSEQueue()
    await StreamAgentUseCase(agent, logger).execute(REQUEST, events)
    return [event async for event in events]


async def test_every_agent_event_is_forwarded_then_the_stream_completes(logger) -> None:
    agent = ScriptedAgent(
        [
            AgentMessageStream.status("Thinking"),
            AgentMessageStream.token("he"),
            AgentMessageStream.reset(),
            AgentMessageStream.token("hello"),
            AgentMessageStream.final("hello", {"iteration": "1"}),
        ]
    )

    emitted = await _run(agent, logger)

    assert [event.type for event in emitted] == [
        MessageStreamType.STATUS,
        MessageStreamType.TOKEN,
        MessageStreamType.RESET,
        MessageStreamType.TOKEN,
        MessageStreamType.FINAL,
        MessageStreamType.COMPLETE,
    ]


async def test_unavailable_agent_yields_a_friendly_error(logger) -> None:
    emitted = await _run(ScriptedAgent([], AgentUnavailableException("upstream down")), logger)

    assert emitted[0].type is MessageStreamType.ERROR
    assert emitted[0].content == AGENT_UNAVAILABLE_MESSAGE


async def test_a_failure_mid_stream_keeps_earlier_events_and_still_completes(logger) -> None:
    emitted = await _run(
        ScriptedAgent([AgentMessageStream.token("partial")], RuntimeError("boom")), logger
    )

    assert [event.type for event in emitted] == [
        MessageStreamType.TOKEN,
        MessageStreamType.ERROR,
        MessageStreamType.COMPLETE,
    ]
    assert "boom" in emitted[1].content
