from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import (
    unpack_state,
)

CLARIFY: str = "clarify"
PLAN: str = "plan"
ACT: str = "act"
TOOLS: str = "tools"
REFLECT: str = "reflect"
FEEDBACK: str = "feedback"
FINALIZE: str = "finalize"

_NEEDS_A_PLAN: frozenset[Intent] = frozenset({Intent.TASK, Intent.PLAN_REVISION})


class AgentRouter:
    @staticmethod
    def after_context(state: GraphState) -> str:
        context = unpack_state(state).turn.context
        if context is None or context.intent in _NEEDS_A_PLAN:
            return PLAN
        if context.intent is Intent.AMBIGUOUS:
            return CLARIFY
        return ACT

    @staticmethod
    def after_act(state: GraphState) -> str:
        agent_state: AgentState = unpack_state(state)
        turn = agent_state.turn
        draft = turn.scratch.last_assistant()
        if draft is None or not draft.tool_calls:
            return FINALIZE if AgentRouter._is_plain_direct_answer(agent_state) else REFLECT
        if turn.iteration >= agent_state.limits.max_iterations:
            return FINALIZE
        return TOOLS

    @staticmethod
    def after_reflect(state: GraphState) -> str:
        agent_state: AgentState = unpack_state(state)
        turn, limits = agent_state.turn, agent_state.limits
        wants_retry: bool = turn.reflection is not None and turn.reflection.should_retry
        if (
            wants_retry
            and turn.retries < limits.max_retries
            and turn.iteration < limits.max_iterations
        ):
            return FEEDBACK
        return FINALIZE

    @staticmethod
    def _is_plain_direct_answer(agent_state: AgentState) -> bool:
        turn = agent_state.turn
        return (
            turn.context is not None
            and turn.context.intent is Intent.DIRECT
            and not turn.scratch.of_role(Role.TOOL)
        )
