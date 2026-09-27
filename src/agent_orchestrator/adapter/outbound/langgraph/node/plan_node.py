from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.node.node import Node
from agent_orchestrator.adapter.outbound.langgraph.prompt.plan_prompt import (
    plan_approval_message,
    plan_system_prompt,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.plan import Plan
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.adapter.outbound.tool.tool_port import ToolRegistryPort
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream


class PlanNode(Node):
    def __init__(
        self,
        llm: BaseChatModel,
        tool_registry: ToolRegistryPort,
        window: ContextWindow,
        logger: Logger,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry
        self._window = window
        self._logger = logger

    async def __call__(self, state: GraphState) -> GraphState:
        agent_state: AgentState = self._unpack(state)
        self._emit(AgentMessageStream.status("Preparing a plan"))

        context = agent_state.turn.context
        task: str = context.standalone_query if context else agent_state.turn.user_message
        previous: Plan | None = agent_state.pending_plan

        system: str = plan_system_prompt(
            task, self._tool_registry.specifications(), previous.steps if previous else None
        )
        messages: list[BaseMessage] = self._window.build(system, agent_state)
        self._logger.debug("Calling planner LLM to lay down a plan")
        raw = await self._llm.ainvoke(messages)

        plan = Plan(task=task, steps=str(raw.content).strip())
        agent_state.pending_plan = plan
        agent_state.turn.answer = plan_approval_message(plan.steps)
        agent_state.turn.outcome = TurnOutcome.AWAITING_APPROVAL
        return self._pack(agent_state)
