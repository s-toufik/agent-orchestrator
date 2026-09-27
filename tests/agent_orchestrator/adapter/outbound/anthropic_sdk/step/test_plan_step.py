from typing import cast

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.exchange import Exchange
from agent_orchestrator.domain.model.tool_specification import ToolSpecification
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import (
    MemoryAgentStates,
    ScriptedStructuredOutput,
    agent_steps,
    drain,
    profile,
    turn_state,
)

ECHO = ToolSpecification("echo", "Echo the rows")


def _plan_step(output: ScriptedStructuredOutput):
    return agent_steps(output, MemoryAgentStates(), builtin_tools=("Glob",), mcp_tools=[ECHO]).plan


async def test_whatever_the_planner_writes_becomes_the_pending_plan() -> None:
    output = ScriptedStructuredOutput("## Plan\n1. Echo the rows (tool: echo)")
    turn = turn_state(Intent.TASK, TurnMode.PLAN, query="count the rows")
    channel = EventChannel()

    await _plan_step(output).run(profile(), turn, channel)

    steps = "## Plan\n1. Echo the rows (tool: echo)"
    assert turn.agent_state.pending_plan == Plan(task="count the rows", steps=steps)
    assert turn.outcome is TurnOutcome.AWAITING_APPROVAL
    assert turn.answer is not None and turn.answer.startswith(steps)
    assert "Reply **yes**" in turn.answer
    assert [event.content for event in await drain(channel)] == ["Preparing a plan"]


async def test_the_planner_sees_the_task_the_tools_and_the_conversation() -> None:
    output = ScriptedStructuredOutput("1. x")
    turn = turn_state(Intent.TASK, TurnMode.PLAN, query="count the rows", user_message="count")
    turn.agent_state.exchanges = [Exchange(user="hi", assistant="hello")]

    await _plan_step(output).run(profile(), turn, EventChannel())

    system = output.systems[0]
    assert "count the rows" in system
    assert "- echo: Echo the rows" in system
    assert "- Glob: find files by name pattern" in system
    assert cast(list, output.conversations[0]) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "count"},
    ]


async def test_a_revision_shows_the_previous_plan_to_the_planner() -> None:
    output = ScriptedStructuredOutput("1. new")
    turn = turn_state(
        Intent.PLAN_REVISION, TurnMode.PLAN, pending=Plan(task="t", steps="1. OLD STEP")
    )

    await _plan_step(output).run(profile(), turn, EventChannel())

    assert "1. OLD STEP" in output.systems[0]
    assert turn.agent_state.pending_plan is not None
    assert turn.agent_state.pending_plan.steps == "1. new"
