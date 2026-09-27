from anthropic.types import MessageParam

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.plan_prompt import (
    plan_approval_message,
    plan_system_prompt,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.structured_output import (
    StructuredOutput,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.step.tools_step import ToolsStep
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream


class PlanStep:
    def __init__(self, structured_output: StructuredOutput, tools: ToolsStep) -> None:
        self._structured_output = structured_output
        self._tools = tools

    async def run(self, profile: ModelProfile, turn: TurnState, channel: EventChannel) -> None:
        await channel.emit(AgentMessageStream.status("Preparing a plan"))
        agent_state = turn.agent_state
        task: str = turn.context.standalone_query
        previous: Plan | None = agent_state.pending_plan

        steps: str = await self._structured_output.invoke(
            profile=profile,
            system=plan_system_prompt(
                task, self._tools.catalog(), previous.steps if previous else None
            ),
            messages=_conversation(turn),
            max_tokens=profile.max_output_tokens,
        )
        agent_state.pending_plan = Plan(task=task, steps=steps)
        turn.answer = plan_approval_message(steps)
        turn.outcome = TurnOutcome.AWAITING_APPROVAL


def _conversation(turn: TurnState) -> list[MessageParam]:
    messages: list[MessageParam] = []
    for exchange in turn.agent_state.exchanges:
        messages.append({"role": "user", "content": exchange.user})
        messages.append({"role": "assistant", "content": exchange.assistant})
    messages.append({"role": "user", "content": turn.user_message})
    return messages
