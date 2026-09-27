from typing import cast

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.structured_output import (
    StructuredOutput,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.reflect_step import ReflectStep
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import (
    ScriptedStructuredOutput,
    accept,
    drain,
    profile,
    turn_state,
)


async def test_the_judge_sees_conversation_request_plan_evidence_and_draft() -> None:
    output = ScriptedStructuredOutput(accept())
    turn = turn_state(
        Intent.PLAN_APPROVAL,
        TurnMode.EXECUTE,
        plan=Plan(task="t", steps="1. APPROVED STEP"),
        user_message="I mean for agents",
        success_criteria=["covers agents"],
    )
    turn.agent_state.record("explain VaR", "VaR is a loss quantile.")
    turn.evidence.append("EVIDENCE" + "x" * 50)
    turn.write("DRAFT")
    channel = EventChannel()

    decision = await ReflectStep(cast(StructuredOutput, output), max_evidence_chars=10).run(
        profile(), turn, channel
    )

    assert decision == accept()
    assert [event.content for event in await drain(channel)] == ["Checking the answer"]
    prompt = output.prompts[0]
    for expected in (
        "User: explain VaR",
        "Assistant: VaR is a loss quantile.",
        "User: I mean for agents",
        "- covers agents",
        "1. APPROVED STEP",
        "EVIDENCExx [...]",
        "DRAFT",
    ):
        assert expected in prompt


def test_reflection_is_skipped_only_for_plans_questions_and_direct_answers_without_tools() -> None:
    applies = ReflectStep.applies

    direct = turn_state(Intent.DIRECT, TurnMode.DIRECT)
    assert not applies(direct)
    direct.tools_used = True
    assert applies(direct)

    assert applies(turn_state(Intent.CONTINUATION, TurnMode.DIRECT))
    assert applies(turn_state(Intent.PLAN_APPROVAL, TurnMode.EXECUTE))
    assert not applies(turn_state(Intent.TASK, TurnMode.PLAN))
    assert not applies(turn_state(Intent.AMBIGUOUS, TurnMode.CLARIFY))
