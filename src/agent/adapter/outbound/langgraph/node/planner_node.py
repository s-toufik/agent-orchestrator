import uuid
from collections.abc import Awaitable, Callable
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage

from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.node import Node
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.schema.planner_decision import PlannerDecision
from agent.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent.adapter.outbound.langgraph.service.prompt_service import PromptService
from agent.adapter.outbound.langgraph.service.tool_mapper import to_langchain_tools
from agent.application.port.outbound.tool_port import ToolRegistryPort


class PlannerNode(Node):
    def __init__(
        self,
        llm: BaseChatModel,
        tool_registry: ToolRegistryPort,
        prompt_service: PromptService,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry
        self._prompt_service = prompt_service
        self._on_token = on_token

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        messages: list[BaseMessage] = [
            SystemMessage(content=self._prompt_service.planner_system_prompt()),
            *agent_state.conversation.to_langchain(),
        ]

        bound_llm = self._llm.bind_tools(to_langchain_tools(self._tool_registry.specifications()))
        raw: AIMessage | AIMessageChunk = (
            await self._stream(bound_llm, messages, self._on_token)
            if self._on_token is not None
            else await bound_llm.ainvoke(messages)
        )

        tool_calls: list[ToolCall] = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=call["name"],
                args=call.get("args", {}),
            )
            for call in raw.tool_calls
        ]

        decision = PlannerDecision(
            tool_calls=tool_calls,
            answer=str(raw.content) if not raw.tool_calls else None,
        )

        agent_state.conversation.append(
            ConversationMessage(
                role=Role.ASSISTANT,
                content=decision.answer or "",
                tool_calls=tool_calls,
            )
        )
        agent_state.planner = decision
        agent_state.iteration += 1
        agent_state.last_node = "planner"

        return self._pack(agent_state)

    @staticmethod
    async def _stream(
        bound_llm,
        messages: list[BaseMessage],
        on_token: Callable[[str], Awaitable[None]],
    ) -> AIMessageChunk:
        accumulated: AIMessageChunk = AIMessageChunk(content="")
        async for chunk in bound_llm.astream(messages):
            accumulated = cast(AIMessageChunk, accumulated + chunk)
            if chunk.content:
                await on_token(str(chunk.content))
        return accumulated
