from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.prompt.context_prompt import (
    context_request,
    context_system_prompt,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_context import TurnContext
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import (
    ContextWindow,
    render,
)
from agent_orchestrator.adapter.outbound.langgraph.service.structured_output import (
    invoke_structured,
)
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream

# A pending plan survives only while the user is still discussing it.
_KEEPS_PENDING_PLAN: frozenset[Intent] = frozenset({Intent.PLAN_REVISION, Intent.AMBIGUOUS})


class ContextNode(Node):
    def __init__(self, llm: BaseChatModel, window: ContextWindow, logger: Logger) -> None:
        self._llm = llm
        self._window = window
        self._logger = logger

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)
        self._emit(AgentMessageStream.status("Understanding your request"))

        context: TurnContext = await self._understand(agent_state)
        self._resolve(agent_state, context)
        self._logger.debug(f"Turn context: {agent_state.turn.context}")
        return self._pack(agent_state)

    async def _understand(self, agent_state: AgentState) -> TurnContext:
        message: str = agent_state.turn.user_message
        pending = agent_state.pending_plan
        history: list[BaseMessage] = self._window.recent(agent_state)[:-1]
        messages: list[BaseMessage] = [
            SystemMessage(content=context_system_prompt(TurnContext.model_json_schema())),
            HumanMessage(
                content=context_request(
                    render(history), pending.steps if pending else None, message
                )
            ),
        ]
        self._logger.debug("Calling context LLM")
        parsed: TurnContext | None = await invoke_structured(self._llm, TurnContext, messages)
        if parsed is None:
            self._logger.warning("Context output could not be parsed; treating it as a new task")
            return TurnContext(intent=Intent.TASK, standalone_query=message)
        return parsed

    @staticmethod
    def _resolve(agent_state: AgentState, context: TurnContext) -> None:
        pending = agent_state.pending_plan

        if context.intent is Intent.PLAN_APPROVAL:
            if pending is None:
                context.intent = Intent.DIRECT
            else:
                agent_state.turn.plan = pending
                context.standalone_query = pending.task
        elif context.intent is Intent.AMBIGUOUS and not context.clarification_question:
            context.intent = Intent.TASK

        if context.intent not in _KEEPS_PENDING_PLAN:
            agent_state.pending_plan = None
        agent_state.turn.context = context
