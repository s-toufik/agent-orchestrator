from typing import Any

from langchain_core.exceptions import ModelError

from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state
from agent.domain.exception.agent_unavailable_exception import AgentUnavailableException
from agent.domain.model.agent_message import AgentMessage
from agent.domain.model.agent_request import AgentRequest


class LangAgent:
    """LangGraph-backed implementation of AgentPort."""

    def __init__(self, graphs: dict[str, Any]) -> None:
        self._graphs = graphs

    async def run(self, request: AgentRequest) -> AgentMessage:
        graph = self._graphs[request.model_name]
        config = {"configurable": {"thread_id": request.request_id}}

        state = await self._load_state(graph, config, request.request_id)
        self._append_user_message(state, request.message)

        result: GraphState = await self._invoke(graph, state, config)
        final_state: AgentState = unpack_state(result)

        return AgentMessage(
            session_id=final_state.session_id or request.request_id,
            content=final_state.final_answer or "",
            metadata={
                "iteration": str(final_state.iteration),
                "max_iteration": str(final_state.max_iterations),
            },
        )

    @staticmethod
    async def _invoke(graph: Any, state: AgentState, config: dict) -> GraphState:
        try:
            return await graph.ainvoke(pack_state(state), config=config)
        except ModelError as exception:
            if exception.is_retryable:
                raise AgentUnavailableException(str(exception)) from exception
            raise

    @staticmethod
    async def _load_state(graph: Any, config: dict, session_id: str) -> AgentState:
        snapshot = await graph.aget_state(config)
        if snapshot.values:
            return unpack_state(snapshot.values)
        return AgentState(session_id=session_id)

    @staticmethod
    def _append_user_message(state: AgentState, message: str) -> None:
        state.iteration = 0
        state.question = message
        state.conversation.append(ConversationMessage(role=Role.USER, content=message))
