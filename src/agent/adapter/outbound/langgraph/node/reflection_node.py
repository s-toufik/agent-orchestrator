from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.adapter.outbound.langgraph.node.node import Node
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.conversation_message import ConversationMessage
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.schema.reflection_decision import ReflectionDecision
from agent.adapter.outbound.langgraph.service.prompt_service import PromptService


class ReflectionNode(Node):
    def __init__(self, llm: BaseChatModel, prompt_service: PromptService) -> None:
        self._llm = llm
        self._prompt_service = prompt_service

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        last: ConversationMessage | None = agent_state.conversation.last_assistant()
        answer: str = last.content if last else "(no assistant answer found)"

        question: str = agent_state.question
        if question:
            first_user_question: ConversationMessage | None = agent_state.conversation.first_user()
            question: str = (
                first_user_question.content if first_user_question else "(no user question found)"
            )

        messages: list[BaseMessage] = [
            SystemMessage(
                content=self._prompt_service.reflection_system_prompt(
                    ReflectionDecision.model_json_schema()
                )
            ),
            HumanMessage(
                content=f"User question is:\n\n {question} Assistant answer is: \n\n{answer}"
            ),
        ]

        raw: dict[str, Any] = cast(
            dict[str, Any],
            await self._llm.with_structured_output(ReflectionDecision, include_raw=True).ainvoke(
                messages
            ),
        )

        agent_state.reflection = cast(ReflectionDecision | None, raw.get("parsed"))
        agent_state.last_node = "reflection"
        return self._pack(agent_state)
