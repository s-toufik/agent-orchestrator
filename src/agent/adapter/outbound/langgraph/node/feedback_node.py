from agent.adapter.outbound.langgraph.enum.role import Role
from agent.adapter.outbound.langgraph.node.node import Node
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.service.prompt_service import PromptService


class FeedbackNode(Node):
    def __init__(self, prompt_service: PromptService) -> None:
        self._prompt_service = prompt_service

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        critique: str = (
            agent_state.reflection.critique
            if agent_state.reflection
            else "No specific critique provided."
        )

        agent_state.conversation.append(
            ConversationMessage(
                role=Role.USER,
                content=self._prompt_service.feedback_system_prompt(critique),
            )
        )
        agent_state.last_node = "feedback"
        return self._pack(agent_state)
