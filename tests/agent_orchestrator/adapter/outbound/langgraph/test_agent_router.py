import pytest

from agent_orchestrator.adapter.outbound.langgraph.agent_router import (
    ACT,
    CLARIFY,
    FEEDBACK,
    FINALIZE,
    PLAN,
    REFLECT,
    TOOLS,
    AgentRouter,
)
from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent
from agent_orchestrator.adapter.outbound.langgraph.enum.role import Role
from agent_orchestrator.adapter.outbound.langgraph.schema.agent_limits import AgentLimits
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation_message import (
    ConversationMessage,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.tool_call import ToolCall
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.state_serialization import pack_state
from tests.agent_orchestrator.adapter.outbound.langgraph.fakes import accept, context, retry


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (Intent.TASK, PLAN),
        (Intent.PLAN_REVISION, PLAN),
        (Intent.PLAN_APPROVAL, ACT),
        (Intent.CONTINUATION, ACT),
        (Intent.DIRECT, ACT),
        (Intent.AMBIGUOUS, CLARIFY),
    ],
)
def test_after_context_routes_on_intent(intent: Intent, expected: str) -> None:
    state = AgentState(turn=TurnState(context=context(intent)))

    assert AgentRouter.after_context(pack_state(state)) == expected


def test_after_context_plans_when_there_is_no_context() -> None:
    assert AgentRouter.after_context(pack_state(AgentState())) == PLAN


def _acting(
    iteration: int,
    with_tool_calls: bool,
    intent: Intent = Intent.TASK,
    used_tools: bool = False,
) -> AgentState:
    calls = [ToolCall(id="1", name="t")] if with_tool_calls else []
    tool_results = [ConversationMessage(role=Role.TOOL, content="r")] if used_tools else []
    draft = ConversationMessage(role=Role.ASSISTANT, content="", tool_calls=calls)
    return AgentState(
        limits=AgentLimits(max_iterations=3),
        turn=TurnState(
            context=context(intent),
            iteration=iteration,
            scratch=Conversation(messages=[*tool_results, draft]),
        ),
    )


def test_after_act_runs_the_tools_that_were_asked_for() -> None:
    assert AgentRouter.after_act(pack_state(_acting(1, with_tool_calls=True))) == TOOLS


def test_after_act_reflects_on_a_plain_answer() -> None:
    assert AgentRouter.after_act(pack_state(_acting(1, with_tool_calls=False))) == REFLECT


def test_after_act_skips_reflection_for_a_direct_answer_without_tools() -> None:
    state = _acting(1, with_tool_calls=False, intent=Intent.DIRECT)

    assert AgentRouter.after_act(pack_state(state)) == FINALIZE


def test_after_act_still_reflects_on_a_direct_answer_that_used_tools() -> None:
    state = _acting(2, with_tool_calls=False, intent=Intent.DIRECT, used_tools=True)

    assert AgentRouter.after_act(pack_state(state)) == REFLECT


def test_after_act_reflects_on_a_continuation() -> None:
    state = _acting(1, with_tool_calls=False, intent=Intent.CONTINUATION)

    assert AgentRouter.after_act(pack_state(state)) == REFLECT


def test_after_act_finalizes_when_the_step_budget_is_spent() -> None:
    assert AgentRouter.after_act(pack_state(_acting(3, with_tool_calls=True))) == FINALIZE


def _reflected(
    decision: ReflectionDecision | None, retries: int = 0, iteration: int = 1
) -> AgentState:
    return AgentState(
        limits=AgentLimits(max_iterations=5, max_retries=2),
        turn=TurnState(reflection=decision, retries=retries, iteration=iteration),
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (_reflected(retry()), FEEDBACK),
        (_reflected(accept()), FINALIZE),
        (_reflected(None), FINALIZE),
        (_reflected(retry(), retries=2), FINALIZE),
        (_reflected(retry(), iteration=5), FINALIZE),
    ],
)
def test_after_reflect_retries_only_within_both_limits(state: AgentState, expected: str) -> None:
    assert AgentRouter.after_reflect(pack_state(state)) == expected
