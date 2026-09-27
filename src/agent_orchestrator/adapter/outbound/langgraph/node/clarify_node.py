from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState


class ClarifyNode(Node):
    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        context = agent_state.turn.context
        agent_state.turn.answer = context.clarification_question if context else None
        agent_state.turn.outcome = TurnOutcome.CLARIFICATION
        return self._pack(agent_state)
