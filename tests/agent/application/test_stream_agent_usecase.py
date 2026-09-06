import pytest

from agent.adapter.outbound.streaming.sse_queue import SSEQueue
from agent.application.use_case.stream_agent_usecase import (
    AGENT_UNAVAILABLE_MESSAGE,
    StreamAgentUseCase,
    on_token,
)
from agent.domain.enum.agent_message_status import MessageStreamType
from agent.domain.exception.agent_unavailable_exception import AgentUnavailableException
from agent.domain.model.agent_message import AgentMessage
from agent.domain.model.agent_request import AgentRequest

REQUEST = AgentRequest(message="hello", model_name="gpt-oss-20b", request_id="r1")


class StubAgent:
    def __init__(self, message: AgentMessage | None = None, error: Exception | None = None) -> None:
        self._message = message
        self._error = error

    async def run(self, request: AgentRequest) -> AgentMessage:
        if self._error:
            raise self._error
        assert self._message is not None
        return self._message


async def drain(queue: SSEQueue) -> list:
    events = []
    while not queue.queue.empty():
        events.append(await queue.queue.get())
    return events


@pytest.fixture
def events() -> SSEQueue:
    return SSEQueue()


async def test_final_answer_then_complete(events: SSEQueue, logger) -> None:
    agent = StubAgent(AgentMessage(session_id="r1", content="42", metadata={"iteration": "2"}))

    await StreamAgentUseCase(agent, logger).execute(REQUEST, events)

    types = [event.type for event in await drain(events)]
    assert types == [MessageStreamType.FINAL, MessageStreamType.COMPLETE]


async def test_unavailable_agent_yields_a_friendly_error(events: SSEQueue, logger) -> None:
    agent = StubAgent(error=AgentUnavailableException("upstream down"))

    await StreamAgentUseCase(agent, logger).execute(REQUEST, events)

    emitted = await drain(events)
    assert emitted[0].type is MessageStreamType.ERROR
    assert emitted[0].content == AGENT_UNAVAILABLE_MESSAGE


async def test_stream_always_completes_even_on_failure(events: SSEQueue, logger) -> None:
    agent = StubAgent(error=RuntimeError("boom"))

    await StreamAgentUseCase(agent, logger).execute(REQUEST, events)

    assert (await drain(events))[-1].type is MessageStreamType.COMPLETE


async def test_tokens_reach_the_queue_of_the_current_request(events: SSEQueue, logger) -> None:
    class StreamingAgent:
        async def run(self, request: AgentRequest) -> AgentMessage:
            await on_token("he")
            await on_token("llo")
            return AgentMessage(session_id="r1", content="hello")

    await StreamAgentUseCase(StreamingAgent(), logger).execute(REQUEST, events)

    emitted = await drain(events)
    assert [event.content for event in emitted if event.type is MessageStreamType.TOKEN] == [
        "he",
        "llo",
    ]


async def test_context_is_reset_after_the_use_case(events: SSEQueue, logger) -> None:
    await StreamAgentUseCase(StubAgent(AgentMessage("r1", "x")), logger).execute(REQUEST, events)

    # Outside any request, a stray token must not raise or leak into a queue.
    await on_token("orphan")
