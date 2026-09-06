from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.execution_node import ExecutorNode
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation import Conversation
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state
from agent.domain.exception.unknown_tool_exception import UnknownToolException
from agent.domain.model.tool_invocation import ToolInvocation
from agent.domain.model.tool_outcome import ToolOutcome


class StubTool:
    def __init__(self, output: str = "ok", error: Exception | None = None) -> None:
        self._output = output
        self._error = error
        self.seen: list[ToolInvocation] = []

    async def invoke(self, invocation: ToolInvocation) -> ToolOutcome:
        self.seen.append(invocation)
        if self._error:
            raise self._error
        return ToolOutcome(
            invocation_id=invocation.id, tool_name=invocation.name, output=self._output
        )


class StubRegistry:
    def __init__(self, tools: dict[str, StubTool]) -> None:
        self._tools = tools

    def get(self, name: str):
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolException(f"Unknown tool: '{name}'.")
        return tool

    def specifications(self) -> list:
        return []


def _last(conversation: Conversation) -> ConversationMessage:
    message = conversation.last()
    assert message is not None
    return message


def _state_with_calls(*calls: ToolCall) -> AgentState:
    return AgentState(
        conversation=Conversation(
            [ConversationMessage(role=Role.ASSISTANT, content="", tool_calls=list(calls))]
        )
    )


async def test_no_op_when_last_assistant_made_no_tool_calls(logger) -> None:
    node = ExecutorNode(StubRegistry({}), logger)
    state = AgentState(
        conversation=Conversation([ConversationMessage(role=Role.ASSISTANT, content="hi")])
    )

    result = unpack_state(await node(pack_state(state)))

    assert result.conversation.messages == state.conversation.messages
    assert result.last_node == "executor"


async def test_no_op_when_there_is_no_assistant_message_at_all(logger) -> None:
    node = ExecutorNode(StubRegistry({}), logger)

    result = unpack_state(await node(pack_state(AgentState())))

    assert result.conversation.messages == []
    assert result.last_node == "executor"


async def test_tool_result_becomes_a_tool_message(logger) -> None:
    tool = StubTool(output="[{'n': 1}]")
    node = ExecutorNode(StubRegistry({"run_sql": tool}), logger)
    call = ToolCall(id="call_1", name="run_sql", args={"query": "select 1"})

    result = unpack_state(await node(pack_state(_state_with_calls(call))))

    tool_message = _last(result.conversation)
    assert tool_message.role is Role.TOOL
    assert tool_message.content == "[{'n': 1}]"
    assert tool_message.tool_call_id == "call_1"
    assert tool.seen[0].name == "run_sql"
    assert tool.seen[0].arguments == {"query": "select 1"}


async def test_multiple_tool_calls_all_produce_a_message(logger) -> None:
    tools = {"a": StubTool(output="a-out"), "b": StubTool(output="b-out")}
    node = ExecutorNode(StubRegistry(tools), logger)
    calls = (ToolCall(id="1", name="a", args={}), ToolCall(id="2", name="b", args={}))

    result = unpack_state(await node(pack_state(_state_with_calls(*calls))))

    contents = {
        m.tool_call_id: m.content for m in result.conversation.messages if m.role is Role.TOOL
    }
    assert contents == {"1": "a-out", "2": "b-out"}


async def test_unknown_tool_produces_a_failed_outcome_message(logger) -> None:
    node = ExecutorNode(StubRegistry({}), logger)
    call = ToolCall(id="1", name="missing", args={})

    result = unpack_state(await node(pack_state(_state_with_calls(call))))

    message = _last(result.conversation)
    assert "Unknown tool" in message.content
    assert logger.messages("warning")


async def test_a_raising_tool_produces_a_failed_outcome_message_instead_of_crashing(logger) -> None:
    tool = StubTool(error=RuntimeError("boom"))
    node = ExecutorNode(StubRegistry({"flaky": tool}), logger)
    call = ToolCall(id="1", name="flaky", args={})

    result = unpack_state(await node(pack_state(_state_with_calls(call))))

    assert "boom" in _last(result.conversation).content
    assert logger.messages("error")
