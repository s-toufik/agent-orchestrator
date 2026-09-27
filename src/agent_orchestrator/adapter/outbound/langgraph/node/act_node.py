import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.prompt.act_prompt import act_system_prompt
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.langgraph.service.tool_mapper import to_langchain_tools
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.adapter.outbound.tool.tool_port import ToolRegistryPort
from agent_orchestrator.domain.model.tool_specification import ToolSpecification


class ActNode(Node):
    def __init__(
        self,
        llm: BaseChatModel,
        tool_registry: ToolRegistryPort,
        window: ContextWindow,
        logger: Logger,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry
        self._window = window
        self._logger = logger

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)
        turn = agent_state.turn

        tools: list[ToolSpecification] = self._tool_registry.specifications()
        request: str = turn.context.standalone_query if turn.context else turn.user_message
        system: str = act_system_prompt(request, turn.plan.steps if turn.plan else None, tools)
        messages: list[BaseMessage] = self._window.build(
            system, agent_state, turn.scratch.to_langchain()
        )
        # Providers reject an empty tool list, so an agent with no tools binds none.
        llm = self._llm.bind_tools(to_langchain_tools(tools)) if turn.plan and tools else self._llm
        self._logger.debug(f"Calling planner LLM (act, iteration {turn.iteration + 1})")
        raw = await llm.ainvoke(messages)

        turn.scratch.append(
            ConversationMessage(
                role=Role.ASSISTANT,
                content=str(raw.content),
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=call["name"],
                        args=call.get("args", {}),
                    )
                    for call in getattr(raw, "tool_calls", None) or []
                ],
            )
        )
        turn.iteration += 1
        return self._pack(agent_state)
