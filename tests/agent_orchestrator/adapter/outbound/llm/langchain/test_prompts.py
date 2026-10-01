from agent_orchestrator.adapter.outbound.llm.langchain.prompts import act, clock, plan
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.act import (
    act_feedback,
    act_system_prompt,
    plan_request,
)
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.context import (
    context_request,
    context_system_prompt,
)
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.plan import plan_system_prompt
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.review import (
    reflection_request,
    reflection_system_prompt,
)
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.summary import summary_request
from agent_orchestrator.adapter.outbound.llm.langchain.prompts.tool_catalog import NO_TOOLS
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification

TOOLS = [ToolSpecification("run_sql", "SQL.")]


def test_the_act_prompt_describes_tool_work_only_under_a_plan() -> None:
    assert "approved this plan" in act_system_prompt("q", "1. do", TOOLS)
    assert "cannot run tools" in act_system_prompt("q", None, TOOLS)
    assert "- run_sql: SQL." in act_system_prompt("q", None, TOOLS)
    assert NO_TOOLS in act_system_prompt("q", None, [])


def test_the_plan_prompt_lists_the_tools_and_the_plan_to_revise() -> None:
    prompt = plan_system_prompt("count users", TOOLS, previous_steps="1. old step")

    assert "count users" in prompt and "- run_sql: SQL." in prompt and "1. old step" in prompt
    assert NO_TOOLS in plan_system_prompt("q", [])


def test_prompts_carry_the_current_time(monkeypatch) -> None:
    monkeypatch.setattr(clock, "utc_now", lambda: "2026-01-01 00:00 UTC")
    monkeypatch.setattr(act, "utc_now", lambda: "2026-01-01 00:00 UTC")
    monkeypatch.setattr(plan, "utc_now", lambda: "2026-01-01 00:00 UTC")

    assert "2026-01-01 00:00 UTC" in act_system_prompt("q", None, [])
    assert "2026-01-01 00:00 UTC" in plan_system_prompt("q", [])


def test_structured_prompts_embed_their_output_schema() -> None:
    assert "SCHEMA" in context_system_prompt("SCHEMA")
    assert "SCHEMA" in reflection_system_prompt("SCHEMA")


def test_requests_fall_back_to_none_when_a_part_is_missing() -> None:
    assert "(none)" in context_request("(none)", None, "hi")
    review = reflection_request("c", "q", [], None, [], "")
    assert review.count("(none)") == 3 and "(empty answer)" in review
    assert "(none)" in summary_request("", "User: hi")


def test_feedback_quotes_the_critique() -> None:
    assert "too short" in act_feedback("too short")


def test_the_context_prompt_checks_for_a_task_before_a_continuation_or_a_direct_answer() -> None:
    prompt = context_system_prompt("SCHEMA")

    assert "Pick the FIRST intent" in prompt
    assert prompt.index("task:") < prompt.index("continuation:") < prompt.index("direct:")
    assert "questions about the assistant itself" in prompt


def test_a_plan_request_is_read_from_any_line_of_the_reply() -> None:
    assert plan_request("NEEDS_PLAN: read it") == "read it"
    assert plan_request("I need the file.\n  NEEDS_PLAN: read it  ") == "read it"
    assert plan_request("Here is the answer.") is None
