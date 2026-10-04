import pytest
from langchain_core.exceptions import ModelError

from agent_orchestrator.adapter.outbound.llm.langchain.reply.streamed_reply import StreamedReply
from agent_orchestrator.adapter.outbound.llm.langchain.reply.whole_reply import WholeReply
from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from tests.agent_orchestrator.adapter.outbound.llm.langchain.fakes import (
    FakeLLM,
    text,
    tool_request,
)
from tests.agent_orchestrator.application.fakes import RecordingEvents


class _Unavailable(ModelError):
    is_retryable = True


async def test_a_whole_reply_is_read_in_one_call() -> None:
    reply = await WholeReply().read(FakeLLM(replies=[text("a b")]), [])  # ty: ignore[invalid-argument-type]

    assert reply.content == "a b"


async def test_a_streamed_reply_publishes_each_piece_and_returns_the_whole_reply() -> None:
    events = RecordingEvents()

    reply = await StreamedReply(events).read(FakeLLM(replies=[text("a b c")]), [])  # ty: ignore[invalid-argument-type]

    assert reply.content == "a b c"
    assert events.published == [
        TurnEvent.answer_delta("a "),
        TurnEvent.answer_delta("b "),
        TurnEvent.answer_delta("c"),
    ]


async def test_a_streamed_tool_call_is_returned_but_never_published() -> None:
    events = RecordingEvents()
    llm = FakeLLM(replies=[tool_request("request_plan", reason="read it")])

    reply = await StreamedReply(events).read(llm, [])  # ty: ignore[invalid-argument-type]

    assert [(call["name"], call["args"]) for call in reply.tool_calls] == [  # ty: ignore[unresolved-attribute]
        ("request_plan", {"reason": "read it"})
    ]
    assert events.published == []


@pytest.mark.parametrize("reader", [WholeReply(), StreamedReply(RecordingEvents())])
async def test_a_retryable_model_error_means_the_agent_is_unavailable(reader) -> None:
    with pytest.raises(AgentUnavailableException):
        await reader.read(FakeLLM(replies=[_Unavailable("503")]), [])
