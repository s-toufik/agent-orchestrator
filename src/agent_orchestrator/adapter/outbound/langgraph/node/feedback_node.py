from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.prompt.act_prompt import act_feedback
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState


class FeedbackNode(Node):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)
        turn = agent_state.turn

        critique: str = turn.reflection.critique if turn.reflection else "No critique provided."
        turn.scratch.append(ConversationMessage(role=Role.USER, content=act_feedback(critique)))
        turn.retries += 1
        self._logger.debug(f"Retry {turn.retries} with critique: {critique}")
        return self._pack(agent_state)
