from typing import Any

from claude_agent_sdk import HookContext, HookJSONOutput
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_models import TurnModels
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.feedback_step import FeedbackStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.reflect_step import ReflectStep
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.tools_step import ToolsStep


class TurnControls:
    def __init__(
        self,
        turn: TurnState,
        channel: EventChannel,
        models: TurnModels,
        tools: ToolsStep,
        reflect: ReflectStep,
        feedback: FeedbackStep,
        logger: Logger,
    ) -> None:
        self._turn = turn
        self._channel = channel
        self._models = models
        self._tools = tools
        self._reflect = reflect
        self._feedback = feedback
        self._logger = logger

    @property
    def turn(self) -> TurnState:
        return self._turn

    async def on_pre_tool_use(
        self, hook_input: Any, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        # Permissions are decided natively by the CLI's rules; this only reports progress.
        await self._tools.announce(self._turn, hook_input.get("tool_name", ""), self._channel)
        return {}

    async def on_stop(
        self, hook_input: Any, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        turn = self._turn
        if not self._reflect.applies(turn) or turn.retries >= self._models.act.limits.max_retries:
            return {}
        try:
            turn.reflection = await self._reflect.run(self._models.reflection, turn, self._channel)
        except Exception as exception:
            # A broken judge must not block the user's answer.
            self._logger.warning(f"Reflection failed, accepting the draft: {exception}")
            turn.reflection = None
        decision = turn.reflection
        if decision is None or not decision.should_retry:
            return {}
        self._logger.debug(f"Retry {turn.retries + 1} with critique: {decision.critique}")
        return await self._feedback.run(turn, decision, self._channel)
