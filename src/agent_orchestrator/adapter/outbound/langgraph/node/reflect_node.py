from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.prompt.reflection_prompt import (
    reflection_request,
    reflection_system_prompt,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import render
from agent_orchestrator.adapter.outbound.langgraph.service.structured_output import (
    invoke_structured,
)
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream

# Previous user message, previous answer, latest user message.
_RECENT_MESSAGES: int = 3


class ReflectNode(Node):
    def __init__(
        self,
        llm: BaseChatModel,
        logger: Logger,
        max_evidence_chars: int = 2_000,
        max_message_chars: int = 4_000,
    ) -> None:
        self._llm = llm
        self._logger = logger
        self._max_evidence_chars = max_evidence_chars
        self._max_message_chars = max_message_chars

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)
        turn = agent_state.turn
        self._emit(AgentMessageStream.status("Checking the answer"))

        draft = turn.scratch.last_assistant()
        context = turn.context
        messages: list[BaseMessage] = [
            SystemMessage(content=reflection_system_prompt(ReflectionDecision.model_json_schema())),
            HumanMessage(
                content=reflection_request(
                    conversation=self._recent_conversation(agent_state),
                    request=context.standalone_query if context else turn.user_message,
                    criteria=context.success_criteria if context else [],
                    plan=turn.plan.steps if turn.plan else None,
                    evidence=[
                        self._clip(message.content, self._max_evidence_chars)
                        for message in turn.scratch.of_role(Role.TOOL)
                    ],
                    answer=draft.content if draft else "",
                )
            ),
        ]
        self._logger.debug("Calling reflection LLM")
        decision = await invoke_structured(self._llm, ReflectionDecision, messages)
        if decision is None:
            # A broken judge must not block a user's answer: accept and log it.
            self._logger.warning("Reflection output could not be parsed; accepting the draft")
        turn.reflection = decision
        return self._pack(agent_state)

    def _recent_conversation(self, agent_state: AgentState) -> str:
        recent = [
            message.model_copy(
                update={"content": self._clip(message.content, self._max_message_chars)}
            )
            for message in agent_state.transcript.messages[-_RECENT_MESSAGES:]
        ]
        return render(Conversation(messages=recent).to_langchain())

    @staticmethod
    def _clip(content: str, limit: int) -> str:
        return content if len(content) <= limit else content[:limit] + " [...]"
