from typing import Any

import anthropic
import httpx2
import pytest

from agent_orchestrator.adapter.outbound.anthropic_sdk.anthropic_sdk_agent import (
    AnthropicSdkAgent,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_roles import ModelRoles
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.finalize_step import BUDGET_EXHAUSTED
from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import (
    SESSION_ID,
    FakeQuery,
    MemoryAgentStates,
    ScriptedStructuredOutput,
    accept,
    agent_steps,
    context,
    failed_call,
    profile,
    result,
    retry,
    stop,
    text_delta,
    tool_result,
    tool_use_start,
    use_tool,
)

PROFILE = profile(max_iterations=6, max_retries=1)
ECHO = "mcp__toolbox__echo"
Status = MessageStreamType


class Harness:
    def __init__(
        self,
        logger,
        *completions: Any,
        turns: list[list[Any]],
        roles: ModelRoles | None = None,
    ) -> None:
        self.structured_output = ScriptedStructuredOutput(*completions)
        self.query = FakeQuery(*turns)
        self.states = MemoryAgentStates()
        self.agent = AnthropicSdkAgent(
            models={"qwen3-8b": PROFILE},
            steps=agent_steps(self.structured_output, self.states, self.query),
            states=self.states,
            logger=logger,
            roles=roles,
        )
        self.events: list[AgentMessageStream] = []

    async def say(self, message: str, model: str = "qwen3-8b") -> AgentMessageStream:
        request = AgentRequest(message=message, model_name=model, request_id="conv-1")
        self.events = [event async for event in self.agent.stream(request)]
        return self.events[-1]

    def types(self) -> list[MessageStreamType]:
        return [event.type for event in self.events]


async def test_a_task_is_planned_then_approved_then_executed_with_tools(logger) -> None:
    harness = Harness(
        logger,
        context(Intent.TASK, "count the rows"),
        "1. Echo the rows (tool: echo)",
        context(Intent.PLAN_APPROVAL, "yes"),
        accept(),
        turns=[
            [
                use_tool(ECHO),
                tool_result("42 rows"),
                text_delta("There are 42 rows."),
                stop([]),
                result(num_turns=2),
            ],
        ],
    )

    proposal = await harness.say("count the rows")

    assert proposal.metadata == {
        "iteration": "0",
        "max_iteration": "6",
        "outcome": "awaiting_approval",
    }
    assert proposal.content.startswith("1. Echo the rows")
    assert "Reply **yes**" in proposal.content
    assert harness.types() == [Status.STATUS, Status.STATUS, Status.TOKEN, Status.FINAL]
    assert harness.events[1].content == "Preparing a plan"
    assert harness.query.options == []
    saved = harness.states.saved["conv-1"]
    assert saved.pending_plan is not None
    assert saved.pending_plan.task == "count the rows"
    assert saved.session_id == "conv-1"
    assert saved.sdk_session_id is None

    answer = await harness.say("yes")

    assert answer.content == "There are 42 rows."
    assert answer.metadata == {"iteration": "2", "max_iteration": "6", "outcome": "answered"}
    assert "Running echo" in [e.content for e in harness.events if e.type is Status.STATUS]
    system_prompt = str(harness.query.options[0].system_prompt)
    assert "1. Echo the rows" in system_prompt
    assert "Request: count the rows" in system_prompt
    assert "42 rows" in harness.structured_output.prompts[-1]
    saved = harness.states.saved["conv-1"]
    assert saved.pending_plan is None
    assert saved.sdk_session_id == SESSION_ID
    assert [exchange.user for exchange in saved.exchanges] == ["count the rows", "yes"]


async def test_any_planner_output_is_a_plan_so_no_model_can_skip_approval(logger) -> None:
    harness = Harness(logger, context(Intent.TASK, "count the rows"), "I would echo.", turns=[])

    proposal = await harness.say("count the rows")

    assert proposal.metadata["outcome"] == "awaiting_approval"
    assert proposal.content.startswith("I would echo.")
    assert harness.states.saved["conv-1"].pending_plan is not None


async def test_a_clarification_is_asked_without_calling_a_model(logger) -> None:
    harness = Harness(
        logger,
        context(Intent.AMBIGUOUS, "?", clarification_question="Which desk?"),
        turns=[],
    )

    answer = await harness.say("the numbers")

    assert answer.content == "Which desk?"
    assert answer.metadata["outcome"] == "clarification"
    assert harness.types() == [Status.STATUS, Status.TOKEN, Status.FINAL]
    assert harness.query.options == []


async def test_each_role_uses_its_configured_model_or_the_selected_one(logger) -> None:
    harness = Harness(
        logger,
        context(Intent.TASK, "count the rows"),
        "1. Echo",
        context(Intent.PLAN_APPROVAL, "yes"),
        accept(),
        turns=[[use_tool(ECHO), text_delta("42."), stop([]), result()]],
        roles=ModelRoles(context=profile(name="small-context"), reflection=profile(name="judge")),
    )

    await harness.say("count the rows")
    await harness.say("yes")

    assert harness.structured_output.models == [
        "small-context",
        "qwen3-8b",
        "small-context",
        "judge",
    ]
    assert harness.query.options[0].model == "qwen3-8b"


async def test_tools_are_refused_before_approval(logger) -> None:
    harness = Harness(
        logger,
        context(Intent.DIRECT, "hello"),
        turns=[[use_tool(ECHO), text_delta("Hi!"), stop([]), result()]],
    )

    answer = await harness.say("hello")

    assert answer.content == "Hi!"
    assert answer.metadata["outcome"] == "answered"
    assert Status.STATUS not in harness.types()[1:]
    assert harness.query.options[0].mcp_servers == {}


async def test_a_rejected_draft_is_reset_and_rewritten(logger) -> None:
    harness = Harness(
        logger,
        context(Intent.CONTINUATION, "explain more"),
        retry("add an example"),
        turns=[
            [
                text_delta("Short."),
                stop([text_delta("Longer, with an example.")]),
                result(num_turns=2),
            ]
        ],
    )

    answer = await harness.say("explain more")

    assert answer.content == "Longer, with an example."
    assert answer.metadata["outcome"] == "best_effort"
    assert Status.RESET in harness.types()
    assert harness.states.saved["conv-1"].exchanges[-1].assistant == answer.content


async def test_text_before_a_tool_call_is_not_the_answer(logger) -> None:
    harness = Harness(
        logger,
        context(Intent.PLAN_APPROVAL, "yes"),
        turns=[[text_delta("Let me check. "), tool_use_start(), text_delta("Done."), result()]],
    )
    answer = await harness.say("yes")

    assert answer.content == "Done."
    assert harness.types()[1:4] == [Status.TOKEN, Status.RESET, Status.TOKEN]


async def test_running_out_of_turns_ends_with_a_clear_message(logger) -> None:
    harness = Harness(
        logger,
        context(Intent.DIRECT, "hello"),
        turns=[[result(num_turns=6, subtype="error_max_turns")]],
    )

    answer = await harness.say("hello")

    assert answer.content == BUDGET_EXHAUSTED
    assert answer.metadata["outcome"] == "budget_exhausted"


async def test_an_answer_that_was_never_streamed_is_sent_once(logger) -> None:
    harness = Harness(logger, context(Intent.DIRECT, "hello"), turns=[[result(text="Hello there")]])

    answer = await harness.say("hello")

    assert answer.content == "Hello there"
    assert harness.types() == [Status.STATUS, Status.RESET, Status.TOKEN, Status.FINAL]


async def test_an_unknown_model_is_rejected(logger) -> None:
    harness = Harness(logger, turns=[])

    with pytest.raises(ValueError, match="Unknown model"):
        await harness.say("hello", model="gpt-9")


async def test_upstream_outages_become_agent_unavailable(logger) -> None:
    request = httpx2.Request("POST", "http://gateway/v1/messages")
    outages: list[Any] = [
        anthropic.APIConnectionError(request=request),
        anthropic.InternalServerError(
            "boom", response=httpx2.Response(503, request=request), body=None
        ),
    ]
    for outage in outages:
        with pytest.raises(AgentUnavailableException):
            await Harness(logger, outage, turns=[]).say("hello")

    rate_limited = Harness(
        logger, context(Intent.DIRECT, "hello"), turns=[[failed_call("rate_limit")]]
    )
    with pytest.raises(AgentUnavailableException):
        await rate_limited.say("hello")


async def test_a_failed_run_is_an_error(logger) -> None:
    harness = Harness(
        logger,
        context(Intent.DIRECT, "hello"),
        turns=[[result(subtype="error_during_execution")]],
    )

    with pytest.raises(RuntimeError, match="error_during_execution"):
        await harness.say("hello")
