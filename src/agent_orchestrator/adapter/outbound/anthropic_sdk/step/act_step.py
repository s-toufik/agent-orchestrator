from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    ToolResultBlock,
    UserMessage,
    query,
)

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.options_factory import OptionsFactory
from agent_orchestrator.adapter.outbound.anthropic_sdk.port.query_port import QueryPort
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.act_prompt import act_system_prompt
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.tool_access import ToolAccess
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.tools_step import ToolsStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.turn_controls import TurnControls
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream

_UNAVAILABLE_ERRORS: frozenset[str] = frozenset({"rate_limit", "server_error"})


class ActStep:
    def __init__(
        self,
        options: OptionsFactory,
        tools: ToolsStep,
        run_query: QueryPort = query,
    ) -> None:
        self._options = options
        self._tools = tools
        self._run_query = run_query

    def system_prompt(self, turn: TurnState) -> str:
        plan = turn.plan if turn.mode is TurnMode.EXECUTE else None
        return act_system_prompt(
            turn.context.standalone_query,
            plan.steps if plan else None,
            self._tools.catalog(),
            self._tools.guidance() if plan else None,
        )

    def access(self, turn: TurnState) -> ToolAccess:
        return self._tools.access(turn)

    async def run(
        self,
        turn: TurnState,
        controls: TurnControls,
        profile: ModelProfile,
        request_id: str,
        channel: EventChannel,
    ) -> ResultMessage:
        result: ResultMessage | None = None
        with TemporaryDirectory(prefix="claude-config-") as config_dir:
            options = self._options.build(
                turn,
                controls,
                profile,
                request_id,
                Path(config_dir),
                self.system_prompt(turn),
                self.access(turn),
            )
            async for message in self._run_query(prompt=turn.user_message, options=options):
                match message:
                    case StreamEvent() if message.parent_tool_use_id is None:
                        await self._on_stream_event(message.event, turn, channel)
                    case UserMessage() if message.parent_tool_use_id is None:
                        turn.evidence.extend(_tool_results(message))
                    case AssistantMessage() if message.error in _UNAVAILABLE_ERRORS:
                        raise AgentUnavailableException(f"Model call failed: {message.error}")
                    case ResultMessage():
                        result = message
        if result is None:
            raise RuntimeError("The agent finished without a result")
        return result

    @staticmethod
    async def _on_stream_event(
        event: dict[str, Any], turn: TurnState, channel: EventChannel
    ) -> None:
        match event.get("type"):
            case "content_block_start" if event.get("content_block", {}).get("type") == "tool_use":
                # Text written before a tool call is a preamble, not the answer.
                if turn.draft:
                    turn.restart_draft()
                    await channel.emit(AgentMessageStream.reset())
            case "content_block_delta" if event.get("delta", {}).get("type") == "text_delta":
                text: str = event["delta"].get("text", "")
                turn.write(text)
                await channel.emit(AgentMessageStream.token(text))


def _tool_results(message: UserMessage) -> list[str]:
    if isinstance(message.content, str):
        return []
    return [
        block.content if isinstance(block.content, str) else str(block.content)
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.content is not None
    ]
