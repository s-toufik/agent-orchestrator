from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.schema.agent_limits import AgentLimits
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState


class IngestNode(Node):
    def __init__(self, limits: AgentLimits) -> None:
        self._limits = limits

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        agent_state.limits = self._limits
        agent_state.transcript.append(
            ConversationMessage(role=Role.USER, content=agent_state.turn.user_message)
        )
        return self._pack(agent_state)
