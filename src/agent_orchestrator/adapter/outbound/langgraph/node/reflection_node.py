from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.langgraph.service.prompt_service import PromptService


class ReflectionNode(Node):
    def __init__(self, llm: BaseChatModel, prompt_service: PromptService, logger: Logger) -> None:
        self._llm = llm
        self._prompt_service = prompt_service
        self._logger = logger

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        last: ConversationMessage | None = agent_state.conversation.last_assistant()
        answer: str = last.content if last else "(no assistant answer found)"

        question: str = agent_state.current_question()

        messages: list[BaseMessage] = [
            SystemMessage(
                content=self._prompt_service.reflection_system_prompt(
                    ReflectionDecision.model_json_schema()
                )
            ),
            HumanMessage(
                content=f"User question is:\n {question} \n\nAssistant answer is: \n{answer}"
            ),
        ]
        self._logger.debug("Calling reflection LLM")
        self._logger.debug(f"Reflection messages: {messages}")
        raw: dict[str, Any] = cast(
            dict[str, Any],
            await self._llm.with_structured_output(ReflectionDecision, include_raw=True).ainvoke(
                messages
            ),
        )

        agent_state.reflection = cast(ReflectionDecision | None, raw.get("parsed"))
        agent_state.last_node = "reflection"
        return self._pack(agent_state)
