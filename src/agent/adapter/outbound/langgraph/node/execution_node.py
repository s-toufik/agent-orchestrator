import asyncio

from pycraftcore.logger.port import Logger

from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.node import Node
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent.application.port.outbound.tool_port import ToolPort, ToolRegistryPort
from agent.domain.exception.unknown_tool_exception import UnknownToolException
from agent.domain.model.tool_invocation import ToolInvocation
from agent.domain.model.tool_outcome import ToolOutcome


class ExecutorNode(Node):
    def __init__(self, tool_registry: ToolRegistryPort, logger: Logger) -> None:
        self._tool_registry = tool_registry
        self._logger = logger

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        last_assistant: ConversationMessage | None = agent_state.conversation.last_assistant()
        if last_assistant is None or not last_assistant.tool_calls:
            agent_state.last_node = "executor"
            return self._pack(agent_state)

        outcomes: list[ToolOutcome] = await asyncio.gather(
            *[self._run(call) for call in last_assistant.tool_calls]
        )

        for outcome in outcomes:
            agent_state.conversation.append(
                ConversationMessage(
                    role=Role.TOOL,
                    content=outcome.content,
                    tool_call_id=outcome.invocation_id,
                )
            )

        agent_state.last_node = "executor"
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
