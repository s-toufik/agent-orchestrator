from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.act_prompt import act_feedback
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_roles import ModelRoles
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.turn_controls import TurnControls
from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import (
    MemoryAgentStates,
    ScriptedStructuredOutput,
    accept,
    agent_steps,
    drain,
    profile,
    retry,
    turn_state,
)

PROFILE = profile(max_iterations=6, max_retries=1)
STOP_INPUT = {"hook_event_name": "Stop", "stop_hook_active": False}
CONTEXT = {"signal": None}


def _controls(
    turn: TurnState, *decisions, logger=None
) -> tuple[TurnControls, EventChannel, ScriptedStructuredOutput]:
    output = ScriptedStructuredOutput(*decisions)
    steps = agent_steps(output, MemoryAgentStates())
    channel = EventChannel()
    controls = TurnControls(
        turn,
        channel,
        ModelRoles().for_turn(PROFILE),
        steps.tools,
        steps.reflect,
        steps.feedback,
        logger,
    )
    return controls, channel, output


async def test_a_tool_about_to_run_is_announced_only_when_the_mode_allows_tools(logger) -> None:
    executing = turn_state(Intent.PLAN_APPROVAL, TurnMode.EXECUTE)
    controls, channel, _ = _controls(executing, logger=logger)
    direct, direct_channel, _ = _controls(turn_state(Intent.DIRECT), logger=logger)
    pre_tool_use = {"hook_event_name": "PreToolUse", "tool_name": "mcp__toolbox__echo"}

    assert await controls.on_pre_tool_use(pre_tool_use, "t1", CONTEXT) == {}
    assert await direct.on_pre_tool_use(pre_tool_use, "t1", CONTEXT) == {}

    assert [event.content for event in await drain(channel)] == ["Running echo"]
    assert executing.tools_used
    assert await drain(direct_channel) == []


async def test_a_rejected_draft_goes_through_reflect_then_feedback(logger) -> None:
    turn = turn_state(Intent.CONTINUATION)
    controls, channel, _ = _controls(turn, retry("units"), logger=logger)

    output = await controls.on_stop(STOP_INPUT, None, CONTEXT)

    assert output == {"decision": "block", "reason": act_feedback("units")}
    assert turn.reflection == retry("units")
    assert [event.type for event in await drain(channel)] == [
        MessageStreamType.STATUS,
        MessageStreamType.RESET,
        MessageStreamType.STATUS,
    ]


async def test_an_accepted_draft_lets_the_agent_stop(logger) -> None:
    turn = turn_state(Intent.CONTINUATION)
    controls, _, _ = _controls(turn, accept(), logger=logger)

    assert await controls.on_stop(STOP_INPUT, None, CONTEXT) == {}
    assert turn.reflection == accept()


async def test_reflection_is_skipped_when_not_needed_or_out_of_retries(logger) -> None:
    direct, _, output = _controls(turn_state(Intent.DIRECT), logger=logger)
    assert await direct.on_stop(STOP_INPUT, None, CONTEXT) == {}

    spent_turn = turn_state(Intent.CONTINUATION)
    spent_turn.retries = PROFILE.limits.max_retries
    spent, _, spent_output = _controls(spent_turn, logger=logger)
    assert await spent.on_stop(STOP_INPUT, None, CONTEXT) == {}

    assert output.prompts == [] and spent_output.prompts == []


async def test_a_broken_or_unreadable_judge_accepts_the_draft(logger) -> None:
    for decision in (RuntimeError("down"), None):
        turn = turn_state(Intent.CONTINUATION)
        controls, _, _ = _controls(turn, decision, logger=logger)

        assert await controls.on_stop(STOP_INPUT, None, CONTEXT) == {}
        assert turn.reflection is None

    assert logger.messages("warning")
