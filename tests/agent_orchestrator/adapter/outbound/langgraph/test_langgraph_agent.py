import pytest
from langchain_core.exceptions import ModelError

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.langgraph_agent import LangGraphAgent
from agent_orchestrator.adapter.outbound.langgraph.schema.agent_limits import AgentLimits
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import pack_state
from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest


class RetryableModelError(ModelError):
    is_retryable = True


class FatalModelError(ModelError):
    is_retryable = False


class FakeSnapshot:
    def __init__(self, values) -> None:
        self.values = values


class FakeGraph:
    def __init__(
        self,
        snapshot_values=None,
        result: GraphState | None = None,
        custom: list[AgentMessageStream] | None = None,
        error: Exception | None = None,
    ):
        self._snapshot_values = snapshot_values
        self._result = result
        self._custom = custom or []
        self._error = error
        self.invoked_with: GraphState | None = None

    async def aget_state(self, config):
        return FakeSnapshot(self._snapshot_values)

    async def astream(self, state, config, stream_mode):
        assert stream_mode == ["custom", "values"]
        self.invoked_with = state
        if self._error:
            raise self._error
        for event in self._custom:
            yield "custom", event
        if self._result is not None:
            yield "values", self._result


def _result(answer: str, session_id: str = "thread-1", **turn_fields) -> GraphState:
    return pack_state(
        AgentState(session_id=session_id, turn=TurnState(answer=answer, **turn_fields))
    )


def _request(message: str = "hi", request_id: str = "thread-1") -> AgentRequest:
    return AgentRequest(message=message, model_name="m1", request_id=request_id)


async def _stream(
    graph: FakeGraph, request: AgentRequest | None = None
) -> list[AgentMessageStream]:
    return [event async for event in LangGraphAgent({"m1": graph}).stream(request or _request())]


async def test_custom_events_are_forwarded_before_the_final_answer() -> None:
    graph = FakeGraph(
        result=_result("hi there"),
        custom=[AgentMessageStream.status("Thinking"), AgentMessageStream.token("hi there")],
    )

    events = await _stream(graph)

    assert [event.type for event in events] == [
        MessageStreamType.STATUS,
        MessageStreamType.TOKEN,
        MessageStreamType.FINAL,
    ]
    assert events[-1].content == "hi there"


async def test_a_brand_new_thread_is_seeded_with_the_user_message() -> None:
    graph = FakeGraph(result=_result("hi there"))

    await _stream(graph, _request("hello"))

    assert graph.invoked_with is not None
    sent = graph.invoked_with["state"]
    assert sent["session_id"] == "thread-1"
    assert sent["turn"]["user_message"] == "hello"


async def test_a_resumed_thread_starts_a_fresh_turn() -> None:
    previous = AgentState(
        session_id="thread-1",
        transcript=Conversation(messages=[ConversationMessage(role=Role.USER, content="earlier")]),
        turn=TurnState(user_message="earlier", iteration=4, answer="old"),
    )
    graph = FakeGraph(snapshot_values=pack_state(previous), result=_result("second"))

    await _stream(graph, _request("again"))

    assert graph.invoked_with is not None
    sent = graph.invoked_with["state"]
    assert sent["turn"]["iteration"] == 0
    assert sent["turn"]["answer"] is None
    assert sent["turn"]["user_message"] == "again"
    assert [m["content"] for m in sent["transcript"]["messages"]] == ["earlier"]


async def test_the_final_event_carries_iteration_limit_and_outcome() -> None:
    state = AgentState(
        session_id="t",
        limits=AgentLimits(max_iterations=6),
        turn=TurnState(answer="ok", iteration=3, outcome=TurnOutcome.ANSWERED),
    )

    events = await _stream(FakeGraph(result=pack_state(state)))

    assert events[-1].metadata == {"iteration": "3", "max_iteration": "6", "outcome": "answered"}


async def test_a_retryable_model_error_becomes_agent_unavailable() -> None:
    with pytest.raises(AgentUnavailableException):
        await _stream(FakeGraph(error=RetryableModelError("down")))


async def test_a_non_retryable_model_error_propagates_unchanged() -> None:
    with pytest.raises(FatalModelError):
        await _stream(FakeGraph(error=FatalModelError("bad request")))


async def test_a_graph_that_yields_no_state_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="without producing a state"):
        await _stream(FakeGraph())
