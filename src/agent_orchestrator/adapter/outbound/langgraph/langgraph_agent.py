from collections.abc import AsyncIterator
from typing import Any

from langchain_core.exceptions import ModelError

from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest


class LangGraphAgent:
    def __init__(self, graphs: dict[str, Any]) -> None:
        self._graphs = graphs

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentMessageStream]:
        graph = self._graphs[request.model_name]
        config = {"configurable": {"thread_id": request.request_id}}

        state = await self._load_state(graph, config, request.request_id)
        state.turn = TurnState(user_message=request.message)

        result: GraphState | None = None
        try:
            async for mode, chunk in graph.astream(
                pack_state(state), config=config, stream_mode=["custom", "values"]
            ):
                if mode == "custom":
                    yield chunk
                else:
                    result = chunk
        except ModelError as exception:
            if exception.is_retryable:
                raise AgentUnavailableException(str(exception)) from exception
            raise

        if result is None:
            raise RuntimeError("The agent graph finished without producing a state")
        yield self._final(unpack_state(result))

    @staticmethod
    def _final(state: AgentState) -> AgentMessageStream:
        turn = state.turn
        return AgentMessageStream.final(
            turn.answer or "",
            metadata={
                "iteration": str(turn.iteration),
                "max_iteration": str(state.limits.max_iterations),
                "outcome": str(turn.outcome or ""),
            },
        )

    @staticmethod
    async def _load_state(graph: Any, config: dict, session_id: str) -> AgentState:
        snapshot = await graph.aget_state(config)
        if snapshot.values:
            return unpack_state(snapshot.values)
        return AgentState(session_id=session_id)
