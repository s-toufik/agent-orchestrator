import pytest
from langchain_core.exceptions import ModelError

from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.lang_agent import LangAgent
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state
from agent.domain.exception.agent_unavailable_exception import AgentUnavailableException
from agent.domain.model.agent_request import AgentRequest


class RetryableModelError(ModelError):
    is_retryable = True


class FatalModelError(ModelError):
    is_retryable = False


class FakeSnapshot:
    def __init__(self, values) -> None:
        self.values = values


class FakeGraph:
    def __init__(
        self, snapshot_values=None, invoke_result=None, invoke_error: Exception | None = None
    ):
        self._snapshot_values = snapshot_values
        self._invoke_result = invoke_result
        self._invoke_error = invoke_error
        self.invoked_with: GraphState | None = None

    async def aget_state(self, config):
        return FakeSnapshot(self._snapshot_values)

    async def ainvoke(self, state, config):
        self.invoked_with = state
        if self._invoke_error:
            raise self._invoke_error
        return self._invoke_result


def _final_state_result(final_answer: str, iteration: int = 1) -> GraphState:
    state = AgentState(
        session_id="thread-1",
        final_answer=final_answer,
        iteration=iteration,
        conversation=Conversation([ConversationMessage(role=Role.ASSISTANT, content=final_answer)]),
    )
    return pack_state(state)


async def test_run_seeds_a_brand_new_thread_when_no_snapshot_exists() -> None:
    graph = FakeGraph(snapshot_values=None, invoke_result=_final_state_result("hi there"))
    agent = LangAgent({"m1": graph})

    message = await agent.run(AgentRequest(message="hello", model_name="m1", request_id="thread-1"))

    assert message.content == "hi there"
    assert message.session_id == "thread-1"
    assert graph.invoked_with is not None
    last_message = graph.invoked_with["state"]["conversation"][-1]
    assert last_message["role"] == "user"
    assert last_message["content"] == "hello"


async def test_run_resumes_from_an_existing_snapshot_and_resets_iteration() -> None:
    previous = AgentState(
        session_id="thread-1",
        iteration=4,
        conversation=Conversation([ConversationMessage(role=Role.USER, content="earlier")]),
    )
    graph = FakeGraph(
        snapshot_values=pack_state(previous),
        invoke_result=_final_state_result("second answer"),
    )
    agent = LangAgent({"m1": graph})

    await agent.run(AgentRequest(message="again", model_name="m1", request_id="thread-1"))

    assert graph.invoked_with is not None
    sent = graph.invoked_with["state"]
    assert sent["iteration"] == 0
    assert [m["content"] for m in sent["conversation"]] == ["earlier", "again"]


async def test_metadata_carries_iteration_and_max_iterations() -> None:
    state = AgentState(session_id="t", final_answer="ok", iteration=3, max_iterations=6)
    graph = FakeGraph(invoke_result=pack_state(state))
    agent = LangAgent({"m1": graph})

    message = await agent.run(AgentRequest(message="hi", model_name="m1", request_id="t"))

    assert message.metadata == {"iteration": "3", "max_iteration": "6"}


async def test_a_retryable_model_error_becomes_agent_unavailable() -> None:
    graph = FakeGraph(invoke_error=RetryableModelError("down"))
    agent = LangAgent({"m1": graph})

    with pytest.raises(AgentUnavailableException):
        await agent.run(AgentRequest(message="hi", model_name="m1", request_id="t"))


async def test_a_non_retryable_model_error_propagates_unchanged() -> None:
    graph = FakeGraph(invoke_error=FatalModelError("bad request"))
    agent = LangAgent({"m1": graph})

    with pytest.raises(FatalModelError):
        await agent.run(AgentRequest(message="hi", model_name="m1", request_id="t"))


async def test_session_id_falls_back_to_the_request_id_when_state_never_set_one() -> None:
    state = AgentState(session_id="", final_answer="ok")
    graph = FakeGraph(invoke_result=pack_state(state))
    agent = LangAgent({"m1": graph})

    message = await agent.run(AgentRequest(message="hi", model_name="m1", request_id="fallback-id"))

    assert message.session_id == "fallback-id"
