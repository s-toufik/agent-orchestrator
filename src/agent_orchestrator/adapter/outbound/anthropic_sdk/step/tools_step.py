from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.act_prompt import (
    builtin_tool_purpose,
    builtin_tools_instructions,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.sdk_settings import SdkSettings
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.tool_access import ToolAccess
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.tool_specification import ToolSpecification


class ToolsStep:
    def __init__(
        self, settings: SdkSettings, mcp_tools: list[ToolSpecification] | None = None
    ) -> None:
        self._builtin_tools = list(settings.builtin_tools)
        self._working_directory = settings.working_directory
        self._mcp_servers = [server.name for server in settings.mcp_servers]
        self._mcp_tools = list(mcp_tools or [])

    def catalog(self) -> list[ToolSpecification]:
        # What the model is told it has, in every mode, as in the LangGraph engine.
        return [
            *self._mcp_tools,
            *(ToolSpecification(tool, builtin_tool_purpose(tool)) for tool in self._builtin_tools),
        ]

    def access(self, turn: TurnState) -> ToolAccess:
        # Tools exist only once a plan is approved. The CLI refuses everything the rules
        # do not allow (dontAsk), and built-in tools only inside the working directory.
        if turn.mode is not TurnMode.EXECUTE:
            return ToolAccess()
        return ToolAccess(
            tools=list(self._builtin_tools),
            allowed=[
                *(f"{tool}(/{self._working_directory}/**)" for tool in self._builtin_tools),
                *(f"mcp__{server}__*" for server in self._mcp_servers),
            ],
        )

    def guidance(self) -> str | None:
        if not self._builtin_tools:
            return None
        return builtin_tools_instructions(str(self._working_directory), self._builtin_tools)

    async def announce(self, turn: TurnState, tool_name: str, channel: EventChannel) -> None:
        # Called before the CLI checks its rules: only tools this mode allows are announced.
        if turn.mode is not TurnMode.EXECUTE:
            return
        turn.tools_used = True
        await channel.emit(AgentMessageStream.status(f"Running {_short(tool_name)}"))


def _short(tool_name: str) -> str:
    return tool_name.rsplit("__", 1)[-1]
