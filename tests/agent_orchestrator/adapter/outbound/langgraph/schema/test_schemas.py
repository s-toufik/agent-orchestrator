from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_orchestrator.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction
from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState


def test_agent_state_defaults() -> None:
    state = AgentState()

    assert state.transcript.messages == []
    assert state.summary == ""
    assert state.pending_plan is None
    assert state.limits.max_iterations == 20
    assert state.limits.max_retries == 2
    assert state.turn.scratch.messages == []
    assert state.turn.context is None
    assert state.turn.outcome is None


def test_reflection_decision_should_retry() -> None:
    assert ReflectionDecision(action=ReflectionAction.RETRY, critique="x").should_retry is True
    assert ReflectionDecision(action=ReflectionAction.ACCEPT, critique="x").should_retry is False


def test_tool_call_defaults_empty_args() -> None:
    assert ToolCall(id="1", name="t").args == {}


def test_conversation_append_and_lookup() -> None:
    conversation = Conversation()
    user = ConversationMessage(role=Role.USER, content="hi")
    assistant = ConversationMessage(role=Role.ASSISTANT, content="hello")

    conversation.append(user)
    conversation.append(assistant)

    assert conversation.last() is assistant
    assert conversation.last_assistant() is assistant
    assert conversation.of_role(Role.USER) == [user]


def test_conversation_lookups_are_empty_when_there_are_no_messages() -> None:
    conversation = Conversation()

    assert conversation.last() is None
    assert conversation.last_assistant() is None
    assert conversation.of_role(Role.TOOL) == []


def test_to_langchain_maps_every_role() -> None:
    tool_call = ToolCall(id="call_1", name="run_sql", args={"query": "select 1"})
    conversation = Conversation(
        messages=[
            ConversationMessage(role=Role.SYSTEM, content="sys"),
            ConversationMessage(role=Role.USER, content="hi"),
            ConversationMessage(role=Role.ASSISTANT, content="", tool_calls=[tool_call]),
            ConversationMessage(role=Role.TOOL, content="result", tool_call_id="call_1"),
        ]
    )

    messages = conversation.to_langchain()

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert messages[2].tool_calls[0]["id"] == "call_1"
    assert messages[2].tool_calls[0]["args"] == {"query": "select 1"}
    assert isinstance(messages[3], ToolMessage)
    assert messages[3].tool_call_id == "call_1"


def test_a_tool_message_without_call_id_maps_to_an_empty_string() -> None:
    conversation = Conversation(messages=[ConversationMessage(role=Role.TOOL, content="r")])

    message = conversation.to_langchain()[0]

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == ""
