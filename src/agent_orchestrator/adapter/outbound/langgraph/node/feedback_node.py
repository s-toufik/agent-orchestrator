from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.service.prompt_service import PromptService


class FeedbackNode(Node):
    def __init__(self, prompt_service: PromptService, logger: Logger) -> None:
        self._prompt_service = prompt_service
        self._logger = logger

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        critique: str = (
            agent_state.reflection.critique
            if agent_state.reflection
            else "No specific critique provided."
        )

        last_assistant: ConversationMessage | None = agent_state.conversation.last_assistant()
        question: str = agent_state.current_question()
        answer: str = last_assistant.content if last_assistant else "(no assistant answer found)"

        feedback: ConversationMessage = ConversationMessage(
            role=Role.USER,
            content=self._prompt_service.feedback_system_prompt(question, answer, critique),
        )
        agent_state.conversation.append(feedback)
        self._logger.debug("Calling feedback system")
        self._logger.debug(f"Feedback system: {feedback}")
        agent_state.last_node = "feedback"
        return self._pack(agent_state)
