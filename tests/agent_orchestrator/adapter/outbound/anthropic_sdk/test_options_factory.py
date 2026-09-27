from dataclasses import replace
from pathlib import Path

from claude_agent_sdk import InMemorySessionStore

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.options_factory import OptionsFactory
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.mcp_server import McpServer
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_roles import ModelRoles
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.sdk_settings import SdkSettings
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.tool_access import ToolAccess
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.turn_controls import TurnControls
from tests.agent_orchestrator.adapter.outbound.anthropic_sdk.fakes import (
    MemoryAgentStates,
    ScriptedStructuredOutput,
    agent_steps,
    profile,
    turn_state,
)

PROFILE = profile(max_iterations=7, max_retries=2)
SETTINGS = SdkSettings(
    base_url="http://127.0.0.1:4000",
    auth_token="token",
    working_directory=Path("/work"),
    mcp_servers=[
        McpServer(name="toolbox", url="http://toolbox/mcp", headers={"Authorization": "x"}),
        McpServer(name="external_mcp_a", url="http://a/sse", transport="sse"),
    ],
)


def _options(
    intent: Intent,
    session_id: str | None = None,
    logger=None,
    model: ModelProfile | None = None,
    settings: SdkSettings = SETTINGS,
    access: ToolAccess = ToolAccess(),
):
    mode = {Intent.PLAN_APPROVAL: TurnMode.EXECUTE, Intent.TASK: TurnMode.PLAN}.get(
        intent, TurnMode.DIRECT
    )
    turn = turn_state(intent, mode, query="the request")
    turn.agent_state.sdk_session_id = session_id
    steps = agent_steps(ScriptedStructuredOutput(), MemoryAgentStates())
    controls = TurnControls(
        turn,
        EventChannel(),
        ModelRoles().for_turn(model or PROFILE),
        steps.tools,
        steps.reflect,
        steps.feedback,
        logger,
    )
    store = InMemorySessionStore()
    options = OptionsFactory(settings, store).build(
        turn,
        controls,
        model or PROFILE,
        "req-1",
        Path("/cfg"),
        "THE SYSTEM PROMPT",
        access,
    )
    return options, controls, store


def test_the_agent_is_sandboxed_to_its_mcp_tools(logger) -> None:
    options, controls, store = _options(Intent.DIRECT, logger=logger)

    assert options.tools == []
    assert options.allowed_tools == []
    assert options.setting_sources == []
    assert options.strict_mcp_config is True
    assert options.permission_mode == "dontAsk"
    assert options.can_use_tool is None
    assert options.hooks is not None
    assert options.hooks["PreToolUse"][0].hooks == [controls.on_pre_tool_use]
    assert options.session_store is store
    assert options.cwd == Path("/work")
    assert options.max_turns == 7
    assert options.include_partial_messages is True


def test_mcp_servers_carry_auth_and_the_request_id(logger) -> None:
    options, _, _ = _options(Intent.PLAN_APPROVAL, logger=logger)

    assert options.mcp_servers == {
        "toolbox": {
            "type": "http",
            "url": "http://toolbox/mcp",
            "headers": {"Authorization": "x", "X-Request-ID": "req-1"},
        },
        "external_mcp_a": {
            "type": "sse",
            "url": "http://a/sse",
            "headers": {"X-Request-ID": "req-1"},
        },
    }


def test_mcp_servers_are_connected_only_to_run_an_approved_plan(logger) -> None:
    direct, _, _ = _options(Intent.DIRECT, logger=logger)
    executing, _, _ = _options(Intent.PLAN_APPROVAL, logger=logger)

    assert direct.mcp_servers == {}
    assert isinstance(executing.mcp_servers, dict)
    assert set(executing.mcp_servers) == {"toolbox", "external_mcp_a"}


def test_the_system_prompt_is_rebuilt_every_turn(logger) -> None:
    options, _, _ = _options(Intent.DIRECT, logger=logger)
    prompt = options.system_prompt

    assert isinstance(prompt, dict)
    assert prompt.get("snapshot") is False
    assert prompt.get("prompt") == "THE SYSTEM PROMPT"


def test_the_cli_gets_the_real_model_limits(logger) -> None:
    options, _, _ = _options(Intent.DIRECT, logger=logger)

    assert options.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == str(PROFILE.max_context_tokens)
    assert options.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == str(PROFILE.max_output_tokens)


def test_reasoning_follows_the_model_profile(logger) -> None:
    silent, _, _ = _options(Intent.DIRECT, logger=logger)
    reasoning, _, _ = _options(
        Intent.DIRECT, logger=logger, model=profile(reasoning_effort="medium")
    )

    assert (silent.effort, silent.thinking) == (None, {"type": "disabled"})
    assert (reasoning.effort, reasoning.thinking) == ("medium", None)


def test_every_model_role_points_at_the_gateway_model(logger) -> None:
    options, _, _ = _options(Intent.DIRECT, session_id="s-1", logger=logger)

    assert options.resume == "s-1"
    assert options.model == "qwen3-8b"
    assert options.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == "token"
    assert options.env["CLAUDE_CONFIG_DIR"] == "/cfg"
    for role in ("OPUS", "SONNET", "HAIKU"):
        assert options.env[f"ANTHROPIC_DEFAULT_{role}_MODEL"] == "qwen3-8b"


def test_the_cli_exports_its_telemetry_to_our_collector_only_when_one_is_configured(logger) -> None:
    silent, _, _ = _options(Intent.DIRECT, logger=logger)
    collected, _, _ = _options(
        Intent.DIRECT,
        logger=logger,
        settings=replace(SETTINGS, otlp_endpoint="http://sirius:4317", environment="debug"),
    )

    assert "CLAUDE_CODE_ENABLE_TELEMETRY" not in silent.env
    assert collected.env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert collected.env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://sirius:4317"
    assert collected.env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "grpc"
    assert collected.env["OTEL_RESOURCE_ATTRIBUTES"] == (
        "deployment.environment=debug,agent.session_id=req-1"
    )
    for anthropic_reporting in ("DISABLE_TELEMETRY", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"):
        assert collected.env[anthropic_reporting] == "1"


def test_the_tools_and_rules_of_the_turn_become_the_cli_permissions(logger) -> None:
    access = ToolAccess(tools=["Glob"], allowed=["Glob(//work/**)", "mcp__toolbox__*"])

    options, _, _ = _options(Intent.PLAN_APPROVAL, logger=logger, access=access)

    assert options.tools == ["Glob"]
    assert options.allowed_tools == ["Glob(//work/**)", "mcp__toolbox__*"]
