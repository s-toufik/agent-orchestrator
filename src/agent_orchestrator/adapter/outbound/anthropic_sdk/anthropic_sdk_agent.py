import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

import anthropic
from claude_agent_sdk import ResultMessage
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.port.agent_state_store_port import (
    AgentStateStorePort,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_roles import ModelRoles
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.agent_steps import AgentSteps
from agent_orchestrator.adapter.outbound.anthropic_sdk.turn_controls import TurnControls
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest


class AnthropicSdkAgent:
    def __init__(
        self,
        models: dict[str, ModelProfile],
        steps: AgentSteps,
        states: AgentStateStorePort,
        logger: Logger,
        roles: ModelRoles | None = None,
    ) -> None:
        self._models = models
        self._roles = roles or ModelRoles()
        self._steps = steps
        self._states = states
        self._logger = logger

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentMessageStream]:
        channel = EventChannel()
        producer = asyncio.create_task(self._produce(request, channel))
        try:
            async for event in channel.drain():
                yield event
            await producer
        finally:
            if not producer.done():
                producer.cancel()
                with suppress(asyncio.CancelledError):
                    await producer

    async def _produce(self, request: AgentRequest, channel: EventChannel) -> None:
        try:
            await self._turn(request, channel)
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as exception:
            raise AgentUnavailableException(str(exception)) from exception
        except anthropic.APIStatusError as exception:
            if exception.status_code >= 500:
                raise AgentUnavailableException(str(exception)) from exception
            raise
        finally:
            channel.close()

    async def _turn(self, request: AgentRequest, channel: EventChannel) -> None:
        selected = self._models.get(request.model_name)
        if selected is None:
            raise ValueError(f"Unknown model '{request.model_name}'")
        models = self._roles.for_turn(selected)
        steps = self._steps

        agent_state = await steps.ingest.run(request.request_id)
        turn = await steps.context.run(models.context, agent_state, request.message, channel)
        self._logger.debug(f"[{request.request_id}] {turn.mode} turn: {turn.context}")

        result: ResultMessage | None = None
        match turn.mode:
            case TurnMode.CLARIFY:
                steps.clarify.run(turn)
            case TurnMode.PLAN:
                await steps.plan.run(models.plan, turn, channel)
            case TurnMode.DIRECT | TurnMode.EXECUTE:
                controls = TurnControls(
                    turn,
                    channel,
                    models,
                    steps.tools,
                    steps.reflect,
                    steps.feedback,
                    self._logger,
                )
                result = await steps.act.run(
                    turn, controls, models.act, request.request_id, channel
                )
        answer, outcome = await steps.finalize.run(turn, channel, result)
        steps.summarize.run(agent_state)
        await self._states.save(agent_state)

        await channel.emit(
            AgentMessageStream.final(
                answer,
                metadata={
                    "iteration": str(result.num_turns if result else 0),
                    "max_iteration": str(models.act.limits.max_iterations),
                    "outcome": str(outcome),
                },
            )
        )
