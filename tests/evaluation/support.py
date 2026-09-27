from deepeval.test_case import ToolCall

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState


def tools_called(state: AgentState) -> list[ToolCall]:
    return [
        ToolCall(name=call.name)
        for message in state.turn.scratch.of_role(Role.ASSISTANT)
        for call in message.tool_calls
    ]


def retrieval_context(state: AgentState) -> list[str]:
    return [message.content for message in state.turn.scratch.of_role(Role.TOOL)]
