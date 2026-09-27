import asyncio

from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.adapter.outbound.tool.tool_port import ToolPort, ToolRegistryPort
from agent_orchestrator.domain.exception.unknown_tool_exception import UnknownToolException
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.tool_invocation import ToolInvocation
from agent_orchestrator.domain.model.tool_outcome import ToolOutcome


class ToolsNode(Node):
    def __init__(self, tool_registry: ToolRegistryPort, logger: Logger) -> None:
        self._tool_registry = tool_registry
        self._logger = logger

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        scratch = agent_state.turn.scratch
        last_assistant: ConversationMessage | None = scratch.last_assistant()
        if last_assistant is None or not last_assistant.tool_calls:
            return state

        names: str = ", ".join(call.name for call in last_assistant.tool_calls)
        self._emit(AgentMessageStream.status(f"Running {names}"))
        outcomes: list[ToolOutcome] = await asyncio.gather(
            *[self._run(call) for call in last_assistant.tool_calls]
        )

        for outcome in outcomes:
            scratch.append(
                ConversationMessage(
                    role=Role.TOOL,
                    content=outcome.content,
                    tool_call_id=outcome.invocation_id,
                )
            )

        return self._pack(agent_state)

    async def _run(self, call: ToolCall) -> ToolOutcome:
        invocation = ToolInvocation(id=call.id, name=call.name, arguments=call.args)
        try:
            tool: ToolPort = self._tool_registry.get(call.name)
            return await tool.invoke(invocation)
        except UnknownToolException as exception:
            self._logger.warning(f"[{call.id}] {exception}")
            return ToolOutcome.failure(call.id, call.name, str(exception))
        except Exception as exception:
            self._logger.error(f"[{call.id}] tool '{call.name}' raised: {exception}")
            return ToolOutcome.failure(call.id, call.name, str(exception))
