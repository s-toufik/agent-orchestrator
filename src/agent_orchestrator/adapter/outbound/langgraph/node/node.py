from abc import ABC, abstractmethod

from langgraph.config import get_stream_writer

from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    pack_state,
    unpack_state,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream


class Node(ABC):
    @abstractmethod
    async def __call__(self, state: GraphState) -> GraphState: ...

    @staticmethod
    def _unpack(graph_state: GraphState) -> AgentState:
        return unpack_state(graph_state)

    @staticmethod
    def _pack(agent_state: AgentState) -> GraphState:
        return pack_state(agent_state)

    @staticmethod
    def _emit(event: AgentMessageStream) -> None:
        # Outside a streaming graph run (e.g. a node called directly) there is no writer.
        try:
            writer = get_stream_writer()
        except RuntimeError:
            return
        writer(event)
