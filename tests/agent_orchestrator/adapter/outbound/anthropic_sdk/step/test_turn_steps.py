from pathlib import Path

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.act_prompt import act_feedback
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.mcp_server import McpServer
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.sdk_settings import SdkSettings
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.tool_access import ToolAccess
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.clarify_step import ClarifyStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.feedback_step import FeedbackStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.ingest_step import IngestStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.summarize_step import SummarizeStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.tools_step import ToolsStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState
from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import (
    MemoryAgentStates,
    drain,
    retry,
    turn_state,
)


async def test_ingest_loads_the_saved_state_or_starts_a_new_one() -> None:
    states = MemoryAgentStates()
    await states.save(AgentState(session_id="known", sdk_session_id="s-1"))
    ingest = IngestStep(states)

    assert (await ingest.run("known")).sdk_session_id == "s-1"
    assert await ingest.run("new") == AgentState(session_id="new")


def test_clarify_asks_exactly_the_question_without_a_model_call() -> None:
    turn = turn_state(Intent.AMBIGUOUS, TurnMode.CLARIFY, clarification_question="Which desk?")

    ClarifyStep.run(turn)

    assert (turn.answer, turn.outcome) == ("Which desk?", TurnOutcome.CLARIFICATION)


def _tools_step(builtin_tools: tuple[str, ...] = ("Glob", "Grep")) -> ToolsStep:
    return ToolsStep(
        SdkSettings(
            base_url="http://gw",
            auth_token="t",
            working_directory=Path("/data/working_directory"),
            builtin_tools=builtin_tools,
            mcp_servers=[McpServer(name="toolbox", url="http://toolbox/mcp")],
        )
    )


def test_tools_exist_only_once_a_plan_is_approved() -> None:
    step = _tools_step()

    for mode in (TurnMode.PLAN, TurnMode.DIRECT, TurnMode.CLARIFY):
        assert step.access(turn_state(mode=mode)) == ToolAccess()

    access = step.access(turn_state(mode=TurnMode.EXECUTE))
    assert access.tools == ["Glob", "Grep"]
    assert access.allowed == [
        "Glob(//data/working_directory/**)",
        "Grep(//data/working_directory/**)",
        "mcp__toolbox__*",
    ]


def test_without_built_in_tools_only_mcp_tools_are_allowed() -> None:
    access = _tools_step(builtin_tools=()).access(turn_state(mode=TurnMode.EXECUTE))

    assert access == ToolAccess(tools=[], allowed=["mcp__toolbox__*"])
    assert _tools_step(builtin_tools=()).guidance() is None


def test_the_guidance_names_the_working_directory_and_each_tool() -> None:
    guidance = _tools_step().guidance()

    assert guidance is not None
    assert "/data/working_directory" in guidance
    assert "- Glob: find files by name pattern" in guidance
    assert "- Grep: search inside files" in guidance


async def test_a_tool_is_announced_only_in_execute_mode() -> None:
    executing, channel = turn_state(mode=TurnMode.EXECUTE), EventChannel()
    await _tools_step().announce(executing, "mcp__toolbox__echo", channel)
    planning, quiet = turn_state(mode=TurnMode.PLAN), EventChannel()
    await _tools_step().announce(planning, "mcp__toolbox__echo", quiet)

    assert [event.content for event in await drain(channel)] == ["Running echo"]
    assert executing.tools_used and not planning.tools_used
    assert await drain(quiet) == []


async def test_feedback_resets_the_draft_and_tells_the_agent_to_continue() -> None:
    turn, channel = turn_state(Intent.CONTINUATION), EventChannel()
    turn.write("draft")

    output = await FeedbackStep().run(turn, retry("add units"), channel)

    assert output == {"decision": "block", "reason": act_feedback("add units")}
    assert turn.retries == 1
    assert turn.draft == ""
    assert [event.type for event in await drain(channel)] == [
        MessageStreamType.RESET,
        MessageStreamType.STATUS,
    ]


def test_summarize_keeps_only_the_latest_exchanges() -> None:
    state = AgentState(session_id="c")
    for i in range(5):
        state.record(f"q{i}", f"a{i}")

    SummarizeStep(max_exchanges=2).run(state)

    assert [exchange.user for exchange in state.exchanges] == ["q3", "q4"]
