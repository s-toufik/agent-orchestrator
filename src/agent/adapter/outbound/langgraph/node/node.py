from abc import ABC, abstractmethod

from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state, unpack_state


class Node(ABC):
    @abstractmethod
    async def __call__(self, state: GraphState) -> GraphState | str: ...

    @staticmethod
    def _unpack(graph_state: GraphState) -> AgentState:
        return unpack_state(graph_state)

    @staticmethod
    def _pack(agent_state: AgentState) -> GraphState:
        return pack_state(agent_state)
