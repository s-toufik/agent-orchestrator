from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.adapter.outbound.langgraph.enum.reflection_action import ReflectionAction
from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.planner_decision import PlannerDecision
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision
from agent.adapter.outbound.langgraph.schema.tool_call import ToolCall


def test_agent_state_defaults() -> None:
    state = AgentState()

    assert state.conversation.messages == []
    assert state.planner is None
    assert state.reflection is None
    assert state.last_node == ""
    assert state.session_id == ""
    assert state.iteration == 0
    assert state.max_iterations == 20
    assert state.final_answer is None


def test_planner_decision_wants_tools_only_with_calls() -> None:
    assert PlannerDecision(tool_calls=[], answer="done").wants_tools is False
    call = ToolCall(id="1", name="t", args={})
    assert PlannerDecision(tool_calls=[call]).wants_tools is True


def test_reflection_decision_should_retry() -> None:
    assert ReflectionDecision(action=ReflectionAction.RETRY, critique="x").should_retry is True
    assert ReflectionDecision(action=ReflectionAction.ACCEPT, critique="x").should_retry is False


def test_tool_call_defaults_empty_args() -> None:
    call = ToolCall(id="1", name="t")
    assert call.args == {}


def test_conversation_append_and_lookup() -> None:
    conversation = Conversation()
    user = ConversationMessage(role=Role.USER, content="hi")
    assistant = ConversationMessage(role=Role.ASSISTANT, content="hello")

    conversation.append(user)
    conversation.append(assistant)

    assert conversation.last() is assistant
    assert conversation.first_user() is user
    assert conversation.last_assistant() is assistant


def test_conversation_last_and_first_user_are_none_when_empty() -> None:
    conversation = Conversation()

    assert conversation.last() is None
    assert conversation.first_user() is None
    assert conversation.last_assistant() is None


def test_conversation_copy_is_a_shallow_independent_list() -> None:
    conversation = Conversation([ConversationMessage(role=Role.USER, content="hi")])

    copy = conversation.copy()
    copy.append(ConversationMessage(role=Role.USER, content="second"))

    assert len(conversation.messages) == 1
    assert len(copy.messages) == 2


def test_to_langchain_round_trips_every_role() -> None:
    tool_call = ToolCall(id="call_1", name="run_sql", args={"query": "select 1"})
    conversation = Conversation(
        [
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
    assert messages[2].tool_calls[0]["name"] == "run_sql"
    assert messages[2].tool_calls[0]["args"] == {"query": "select 1"}
    assert isinstance(messages[3], ToolMessage)
    assert messages[3].tool_call_id == "call_1"


def test_to_tool_defaults_missing_call_id_to_empty_string() -> None:
    conversation = Conversation(
        [ConversationMessage(role=Role.TOOL, content="r", tool_call_id=None)]
    )

    message = conversation.to_langchain()[0]

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == ""


def test_from_langchain_round_trips_every_message_type() -> None:
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "run_sql", "args": {"q": 1}}]),
        ToolMessage(content="result", tool_call_id="call_1"),
    ]

    conversation = Conversation.from_langchain(messages)

    assert [m.role for m in conversation.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
    ]
    assert conversation.messages[2].tool_calls == [
        ToolCall(id="call_1", name="run_sql", args={"q": 1})
    ]
    assert conversation.messages[3].tool_call_id == "call_1"


def test_from_langchain_defaults_missing_tool_call_id_to_empty_string() -> None:
    messages = [AIMessage(content="", tool_calls=[{"id": None, "name": "t", "args": {}}])]

    conversation = Conversation.from_langchain(messages)

    assert conversation.messages[0].tool_calls[0].id == ""
