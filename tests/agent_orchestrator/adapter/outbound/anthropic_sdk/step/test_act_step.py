from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.tool_access import ToolAccess
from agent_orchestrator.domain.model.tool_specification import ToolSpecification
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import (
    MemoryAgentStates,
    ScriptedStructuredOutput,
    agent_steps,
    turn_state,
)


def _act(builtin_tools: tuple[str, ...] = ()):
    return agent_steps(
        ScriptedStructuredOutput(),
        MemoryAgentStates(),
        builtin_tools=builtin_tools,
        mcp_tools=[ToolSpecification("echo", "Echo the rows")],
    ).act


def _execute():
    return turn_state(
        Intent.PLAN_APPROVAL,
        TurnMode.EXECUTE,
        plan=Plan(task="t", steps="1. APPROVED"),
        query="count the rows",
    )


def test_the_prompt_carries_the_request_the_tools_and_the_mode() -> None:
    act = _act()
    direct = act.system_prompt(turn_state(Intent.DIRECT, TurnMode.DIRECT, query="what is VaR"))
    execute = act.system_prompt(_execute())

    assert "Request: what is VaR" in direct
    assert "- echo: Echo the rows" in direct
    assert "cannot run tools" in direct
    assert "Request: count the rows" in execute
    assert "1. APPROVED" in execute
    assert "cannot run tools" not in execute


def test_the_prompt_lets_the_model_rely_on_the_earlier_exchanges() -> None:
    # The resumed session already holds them; a small model denies having them otherwise.
    direct = _act().system_prompt(turn_state(Intent.DIRECT, TurnMode.DIRECT))

    assert "your past exchanges with this user" in direct


def test_built_in_tools_are_described_only_when_enabled_and_executing() -> None:
    assert "Built-in file tools" not in _act().system_prompt(_execute())

    act = _act(("Glob", "Grep"))
    described = act.system_prompt(_execute())
    assert "Built-in file tools" in described
    assert "/work" in described
    assert "Built-in file tools" not in act.system_prompt(turn_state(Intent.DIRECT))


def test_tools_are_granted_only_when_executing() -> None:
    act = _act(("Glob",))

    assert act.access(turn_state(Intent.DIRECT)) == ToolAccess()
    assert act.access(_execute()) == ToolAccess(
        tools=["Glob"], allowed=["Glob(//work/**)", "mcp__toolbox__*"]
    )
