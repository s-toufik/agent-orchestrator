from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.prompt.summary_prompt import (
    summary_request,
    summary_system_prompt,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import render
from agent_orchestrator.adapter.outbound.langgraph.service.tokens_service import (
    count_message_tokens,
)
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState


class SummarizeNode(Node):
    def __init__(
        self,
        llm: BaseChatModel,
        logger: Logger,
        trigger_tokens: int,
        keep_last_messages: int = 4,
    ) -> None:
        self._llm = llm
        self._logger = logger
        self._trigger_tokens = trigger_tokens
        self._keep_last_messages = keep_last_messages

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)

        messages = agent_state.transcript.messages
        if len(messages) <= self._keep_last_messages:
            return state
        if count_message_tokens(agent_state.transcript.to_langchain()) <= self._trigger_tokens:
            return state

        older = Conversation(messages=messages[: -self._keep_last_messages])
        try:
            self._logger.debug("Calling summary LLM")
            raw = await self._llm.ainvoke(
                [
                    SystemMessage(content=summary_system_prompt()),
                    HumanMessage(
                        content=summary_request(agent_state.summary, render(older.to_langchain()))
                    ),
                ]
            )
        except Exception as exception:
            # The answer is already delivered; a failed summary only delays compaction.
            self._logger.warning(f"Summary failed, transcript kept as is: {exception}")
            return state

        agent_state.summary = str(raw.content).strip()
        agent_state.transcript = Conversation(messages=messages[-self._keep_last_messages :])
        return self._pack(agent_state)
