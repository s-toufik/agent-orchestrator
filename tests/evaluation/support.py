from deepeval.test_case import ToolCall

from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState


def tools_called(state: AgentState) -> list[ToolCall]:
    """Every tool the agent actually invoked while answering, in call order."""
    return [
        ToolCall(name=call.name)
        for message in state.conversation.messages
        if message.role is Role.ASSISTANT
        for call in message.tool_calls
    ]


def retrieval_context(state: AgentState) -> list[str]:
    """The raw content each tool call returned -- the agent's only source of
    ground truth beyond its own training data."""
    return [message.content for message in state.conversation.messages if message.role is Role.TOOL]
