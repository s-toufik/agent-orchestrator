from typing import cast

import pytest

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.structured_output import (
    StructuredOutput,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.context_step import ContextStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import (
    ScriptedStructuredOutput,
    context,
    drain,
    profile,
)

PLAN = Plan(task="count EQD positions", steps="1. PENDING STEP")


def _state(pending: Plan | None = None) -> AgentState:
    state = AgentState(session_id="c", pending_plan=pending)
    state.record("explain VaR", "VaR is a loss quantile.")
    return state


async def _run(*results, pending: Plan | None = None, message: str = "msg"):
    output = ScriptedStructuredOutput(*results)
    channel = EventChannel()
    turn = await ContextStep(cast(StructuredOutput, output)).run(
        profile(), _state(pending), message, channel
    )
    return turn, output, await drain(channel)


async def test_the_model_sees_history_pending_plan_and_message_and_a_status_is_sent() -> None:
    turn, output, events = await _run(context(Intent.DIRECT, "hi"), pending=PLAN, message="yes?")

    prompt = output.prompts[0]
    assert "User: explain VaR\nAssistant: VaR is a loss quantile." in prompt
    assert "1. PENDING STEP" in prompt
    assert prompt.endswith("yes?")
    assert [event.content for event in events] == ["Understanding your request"]
    assert turn.user_message == "yes?"


async def test_an_unreadable_answer_falls_back_to_a_task() -> None:
    turn, _, _ = await _run(None, message="count users")

    assert turn.context.intent is Intent.TASK
    assert turn.context.standalone_query == "count users"
    assert turn.mode is TurnMode.PLAN


@pytest.mark.parametrize(
    ("intent", "mode"),
    [
        (Intent.TASK, TurnMode.PLAN),
        (Intent.PLAN_REVISION, TurnMode.PLAN),
        (Intent.CONTINUATION, TurnMode.DIRECT),
        (Intent.DIRECT, TurnMode.DIRECT),
    ],
)
async def test_the_intent_decides_the_mode(intent: Intent, mode: TurnMode) -> None:
    turn, _, _ = await _run(context(intent))

    assert turn.mode is mode


async def test_approving_a_pending_plan_executes_it() -> None:
    turn, _, _ = await _run(context(Intent.PLAN_APPROVAL, "yes"), pending=PLAN)

    assert turn.mode is TurnMode.EXECUTE
    assert turn.plan == PLAN
    assert turn.context.standalone_query == PLAN.task
    assert turn.agent_state.pending_plan is None


async def test_approval_without_a_pending_plan_is_answered_directly() -> None:
    turn, _, _ = await _run(context(Intent.PLAN_APPROVAL))

    assert turn.mode is TurnMode.DIRECT
    assert turn.context.intent is Intent.DIRECT


async def test_ambiguity_asks_a_question_only_when_there_is_one() -> None:
    asked, _, _ = await _run(context(Intent.AMBIGUOUS, clarification_question="Which desk?"))
    unclear, _, _ = await _run(context(Intent.AMBIGUOUS))

    assert asked.mode is TurnMode.CLARIFY
    assert unclear.mode is TurnMode.PLAN


@pytest.mark.parametrize(
    ("intent", "kept"),
    [(Intent.PLAN_REVISION, True), (Intent.TASK, False), (Intent.DIRECT, False)],
)
async def test_a_pending_plan_survives_only_while_it_is_discussed(
    intent: Intent, kept: bool
) -> None:
    turn, _, _ = await _run(context(intent), pending=PLAN)

    assert (turn.agent_state.pending_plan is not None) is kept
