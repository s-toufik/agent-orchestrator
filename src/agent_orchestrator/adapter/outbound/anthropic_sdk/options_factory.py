from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, SessionStore
from claude_agent_sdk.types import (
    McpHttpServerConfig,
    McpServerConfig,
    McpSSEServerConfig,
    SystemPromptCustom,
    ThinkingConfigDisabled,
)

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.mcp_server import McpServer
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.sdk_settings import SdkSettings
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.tool_access import ToolAccess
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.turn_controls import TurnControls

STOP_HOOK_TIMEOUT_SECONDS: float = 300.0


class OptionsFactory:
    def __init__(self, settings: SdkSettings, sessions: SessionStore) -> None:
        self._settings = settings
        self._sessions = sessions

    def build(
        self,
        turn: TurnState,
        controls: TurnControls,
        profile: ModelProfile,
        request_id: str,
        config_dir: Path,
        system_prompt: str,
        access: ToolAccess,
    ) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            model=profile.name,
            # snapshot=False: a resumed session would otherwise keep its first turn's prompt.
            system_prompt=SystemPromptCustom(
                type="custom",
                prompt=system_prompt,
                snapshot=False,
            ),
            tools=access.tools,
            allowed_tools=access.allowed,
            mcp_servers=self._mcp_servers(turn, request_id),
            strict_mcp_config=True,
            setting_sources=[],
            # dontAsk: whatever the rules do not allow is refused by the CLI itself.
            permission_mode="dontAsk",
            hooks={
                "PreToolUse": [HookMatcher(hooks=[controls.on_pre_tool_use])],
                "Stop": [HookMatcher(hooks=[controls.on_stop], timeout=STOP_HOOK_TIMEOUT_SECONDS)],
            },
            include_partial_messages=True,
            session_store=self._sessions,
            resume=turn.agent_state.sdk_session_id,
            cwd=self._settings.working_directory,
            env=self._environment(profile, config_dir, request_id),
            max_turns=profile.limits.max_iterations,
            # No effort means the model does not reason: turn thinking off so nothing is sent.
            effort=profile.reasoning_effort,
            thinking=None if profile.reasoning_effort else ThinkingConfigDisabled(type="disabled"),
        )

    def _mcp_servers(self, turn: TurnState, request_id: str) -> dict[str, McpServerConfig]:
        # Outside an approved plan the servers are not even connected, so no model can
        # call a tool there, as in the LangGraph engine where tools are not bound.
        if turn.mode is not TurnMode.EXECUTE:
            return {}
        return {
            server.name: _server_config(server, request_id) for server in self._settings.mcp_servers
        }

    def _environment(
        self, profile: ModelProfile, config_dir: Path, request_id: str
    ) -> dict[str, str]:
        model = profile.name
        return self._telemetry(request_id) | {
            "ANTHROPIC_BASE_URL": self._settings.base_url,
            "ANTHROPIC_AUTH_TOKEN": self._settings.auth_token,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(profile.max_context_tokens),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(profile.max_output_tokens),
            "CLAUDE_CONFIG_DIR": str(config_dir),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_AUTOUPDATER": "1",
        }

    def _telemetry(self, request_id: str) -> dict[str, str]:
        # Anthropic's telemetry stays off; this sends the CLI's OpenTelemetry to our collector.
        # Prompts and responses are redacted by the CLI unless OTEL_LOG_USER_PROMPTS is set.
        if not self._settings.otlp_endpoint:
            return {}
        return {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_METRICS_EXPORTER": "otlp",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
            "OTEL_EXPORTER_OTLP_ENDPOINT": self._settings.otlp_endpoint,
            "OTEL_RESOURCE_ATTRIBUTES": (
                f"deployment.environment={self._settings.environment},agent.session_id={request_id}"
            ),
        }


def _server_config(server: McpServer, request_id: str) -> McpServerConfig:
    headers: dict[str, str] = {**server.headers, "X-Request-ID": request_id}
    if server.transport == "sse":
        return McpSSEServerConfig(type="sse", url=server.url, headers=headers)
    return McpHttpServerConfig(type="http", url=server.url, headers=headers)
