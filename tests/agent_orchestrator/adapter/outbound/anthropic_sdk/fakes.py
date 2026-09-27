from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    InMemorySessionStore,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    UserMessage,
)
from claude_agent_sdk.types import EffortLevel
from pydantic import BaseModel

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.options_factory import OptionsFactory
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.agent_limits import AgentLimits
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.mcp_server import McpServer
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.sdk_settings import SdkSettings
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_context import TurnContext
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.structured_output import (
    StructuredOutput,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.act_step import ActStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.agent_steps import AgentSteps
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.clarify_step import ClarifyStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.context_step import ContextStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.feedback_step import FeedbackStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.finalize_step import FinalizeStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.ingest_step import IngestStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.plan_step import PlanStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.reflect_step import ReflectStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.summarize_step import SummarizeStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.tools_step import ToolsStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.tool_specification import ToolSpecification

SESSION_ID = "11111111-2222-3333-4444-555555555555"


def context(intent: Intent, query: str = "the query", **fields: Any) -> TurnContext:
    return TurnContext(intent=intent, standalone_query=query, **fields)


def accept() -> ReflectionDecision:
    return ReflectionDecision(action="accept")


def retry(critique: str = "too short") -> ReflectionDecision:
    return ReflectionDecision(action="retry", critique=critique)


class ScriptedStructuredOutput:
    def __init__(self, *results: BaseModel | str | None | Exception) -> None:
        self._results = list(results)
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.conversations: list[list[Any]] = []
        self.models: list[str] = []

    async def invoke(
        self, profile: Any, system: str, messages: list[Any], max_tokens: int | None = None
    ) -> Any:
        self.models.append(profile.name)
        self.systems.append(system)
        self.conversations.append(messages)
        return self._next()

    async def invoke_structured(self, profile: Any, system: str, prompt: str, schema: type) -> Any:
        self.models.append(profile.name)
        self.systems.append(system)
        self.prompts.append(prompt)
        return self._next()

    def _next(self) -> Any:
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class MemoryAgentStates:
    def __init__(self) -> None:
        self.saved: dict[str, AgentState] = {}

    async def load(self, session_id: str) -> AgentState | None:
        stored = self.saved.get(session_id)
        return stored.model_copy(deep=True) if stored else None

    async def save(self, agent_state: AgentState) -> None:
        self.saved[agent_state.session_id] = agent_state.model_copy(deep=True)


def text_delta(text: str) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id=SESSION_ID,
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
    )


def tool_use_start() -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id=SESSION_ID,
        event={"type": "content_block_start", "content_block": {"type": "tool_use"}},
    )


def tool_result(content: str) -> UserMessage:
    return UserMessage(content=[ToolResultBlock(tool_use_id="t1", content=content)])


def result(num_turns: int = 1, subtype: str = "success", text: str | None = None) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=subtype != "success",
        num_turns=num_turns,
        session_id=SESSION_ID,
        result=text,
    )


def failed_call(error: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text="")], model="m", error=error)  # ty: ignore[invalid-argument-type]


Step = Callable[[ClaudeAgentOptions], Awaitable[list[Any]]]


class FakeQuery:
    def __init__(self, *turns: list[Any | Step]) -> None:
        self._turns = list(turns)
        self.options: list[ClaudeAgentOptions] = []
        self.prompts: list[str] = []

    async def __call__(self, *, prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Any]:
        self.prompts.append(prompt)
        self.options.append(options)
        for step in self._turns.pop(0):
            messages = await step(options) if callable(step) else [step]
            for message in messages:
                yield message


def permitted(options: ClaudeAgentOptions, name: str, tool_input: dict[str, Any]) -> bool:
    # What the CLI does in dontAsk mode (checked against the real CLI): a built-in tool must
    # be enabled, and some allow rule must match the call.
    if not name.startswith("mcp__") and name not in (options.tools or []):
        return False
    path = str(tool_input.get("path") or tool_input.get("file_path") or "")
    for rule in options.allowed_tools:
        if rule == name or (rule.endswith("*") and name.startswith(rule[:-1])):
            return True
        if rule.startswith(f"{name}(//") and rule.endswith("/**)"):
            if path.startswith(rule[len(name) + 2 : -4] + "/"):
                return True
    return False


def use_tool(name: str, tool_input: dict[str, Any] | None = None) -> Step:
    async def step(options: ClaudeAgentOptions) -> list[Any]:
        arguments = tool_input or {}
        assert options.hooks is not None
        pre_tool_use = options.hooks["PreToolUse"][0].hooks[0]
        await pre_tool_use(
            {"hook_event_name": "PreToolUse", "tool_name": name, "tool_input": arguments},
            "t1",
            {"signal": None},
        )
        if permitted(options, name, arguments):
            return []
        return [tool_result(f"Permission to use {name} has been denied")]

    return step


def controls_of(options: ClaudeAgentOptions) -> Any:
    assert options.hooks is not None
    return getattr(options.hooks["Stop"][0].hooks[0], "__self__")


def stop(then: list[Any]) -> Step:
    async def step(options: ClaudeAgentOptions) -> list[Any]:
        assert options.hooks is not None
        hook = options.hooks["Stop"][0].hooks[0]
        output = await hook(
            {"hook_event_name": "Stop", "stop_hook_active": False}, None, {"signal": None}
        )
        return then if output.get("decision") == "block" else []

    return step


def turn_state(
    intent: Intent = Intent.DIRECT,
    mode: TurnMode = TurnMode.DIRECT,
    pending: Plan | None = None,
    plan: Plan | None = None,
    user_message: str = "msg",
    **context_fields: Any,
) -> TurnState:
    return TurnState(
        user_message=user_message,
        agent_state=AgentState(session_id="c", pending_plan=pending),
        context=context(intent, **context_fields),
        mode=mode,
        plan=plan,
    )


async def drain(channel: EventChannel) -> list[AgentMessageStream]:
    channel.close()
    return [event async for event in channel.drain()]


def agent_steps(
    structured_output: ScriptedStructuredOutput,
    states: MemoryAgentStates,
    run_query: FakeQuery | None = None,
    builtin_tools: tuple[str, ...] = (),
    mcp_tools: list[ToolSpecification] | None = None,
) -> AgentSteps:
    output = cast(StructuredOutput, structured_output)
    settings = SdkSettings(
        base_url="http://gateway",
        auth_token="t",
        working_directory=Path("/work"),
        builtin_tools=builtin_tools,
        mcp_servers=[McpServer(name="toolbox", url="http://toolbox/mcp")],
    )
    tools = ToolsStep(settings, mcp_tools)
    return AgentSteps(
        ingest=IngestStep(states),
        context=ContextStep(output),
        clarify=ClarifyStep(),
        plan=PlanStep(output, tools),
        act=ActStep(
            OptionsFactory(settings, InMemorySessionStore()), tools, run_query or FakeQuery()
        ),
        tools=tools,
        reflect=ReflectStep(output),
        feedback=FeedbackStep(),
        finalize=FinalizeStep(),
        summarize=SummarizeStep(),
    )


def profile(
    max_iterations: int = 6,
    max_retries: int = 1,
    reasoning_effort: EffortLevel | None = None,
    name: str = "qwen3-8b",
) -> ModelProfile:
    return ModelProfile(
        name=name,
        limits=AgentLimits(max_iterations=max_iterations, max_retries=max_retries),
        max_context_tokens=22_000,
        max_output_tokens=4_000,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
    )
