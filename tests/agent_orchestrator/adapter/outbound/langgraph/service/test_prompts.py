from datetime import UTC, datetime

from agent_orchestrator.adapter.outbound.langgraph.prompt import act_prompt, plan_prompt
from agent_orchestrator.adapter.outbound.langgraph.prompt.act_prompt import (
    act_feedback,
    act_system_prompt,
)
from agent_orchestrator.adapter.outbound.langgraph.prompt.context_prompt import (
    context_request,
    context_system_prompt,
)
from agent_orchestrator.adapter.outbound.langgraph.prompt.plan_prompt import (
    plan_approval_message,
    plan_system_prompt,
)
from agent_orchestrator.adapter.outbound.langgraph.prompt.reflection_prompt import (
    reflection_request,
    reflection_system_prompt,
)
from agent_orchestrator.adapter.outbound.langgraph.prompt.summary_prompt import summary_request
from agent_orchestrator.adapter.outbound.langgraph.prompt.tool_catalog import NO_TOOLS
from agent_orchestrator.domain.model.tool_specification import ToolSpecification


def test_context_prompt_embeds_the_schema_and_the_pending_plan() -> None:
    assert "SCHEMA" in context_system_prompt("SCHEMA")
    request = context_request("User: hi", "1. step", "yes")
    assert "1. step" in request
    assert "yes" in request
    assert "(none)" in context_request("User: hi", None, "hello")


def test_plan_prompt_lists_the_tools_and_the_previous_plan_to_revise() -> None:
    tools = [ToolSpecification(name="run_sql", description="SQL.", parameters={})]

    prompt = plan_system_prompt("count users", tools, previous_steps="1. old step")

    assert "count users" in prompt
    assert "- run_sql: SQL." in prompt
    assert "1. old step" in prompt


def test_plan_approval_message_asks_the_user_to_confirm() -> None:
    message = plan_approval_message("## Plan\n1. do")

    assert message.startswith("## Plan")
    assert "yes" in message


def test_act_prompt_only_describes_tool_work_when_a_plan_was_approved() -> None:
    assert "approved this plan" in act_system_prompt("q", "1. do", [])
    assert "cannot run tools" in act_system_prompt("q", None, [])


def test_act_prompt_lets_the_model_rely_on_the_earlier_exchanges() -> None:
    assert "your past exchanges with this user" in act_system_prompt("q", None, [])


def test_act_prompt_always_lists_the_tool_catalog() -> None:
    tools = [ToolSpecification(name="run_sql", description="SQL.", parameters={})]

    assert "- run_sql: SQL." in act_system_prompt("q", None, tools)
    assert "- run_sql: SQL." in act_system_prompt("q", "1. do", tools)


def test_the_catalog_says_so_when_there_are_no_tools() -> None:
    assert NO_TOOLS in act_system_prompt("q", None, [])
    assert NO_TOOLS in plan_system_prompt("q", [])


def test_act_feedback_carries_the_critique() -> None:
    assert "missing units" in act_feedback("missing units")


def test_prompts_state_the_current_utc_time() -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d")

    assert now in act_prompt.act_system_prompt("q", None, [])
    assert now in plan_prompt.plan_system_prompt("q", [])


def test_reflection_prompt_embeds_every_input() -> None:
    assert "SCHEMA" in reflection_system_prompt("SCHEMA")
    request = reflection_request("User: hi", "req", ["crit"], "plan", ["ev"], "ans")
    for part in ("User: hi", "req", "- crit", "plan", "ev", "ans"):
        assert part in request


def test_summary_request_merges_the_existing_summary() -> None:
    assert "old" in summary_request("old", "User: hi")
    assert "(none)" in summary_request("", "User: hi")
