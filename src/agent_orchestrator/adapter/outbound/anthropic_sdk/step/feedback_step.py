from claude_agent_sdk import HookJSONOutput

from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.act_prompt import act_feedback
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream


class FeedbackStep:
    async def run(
        self, turn: TurnState, decision: ReflectionDecision, channel: EventChannel
    ) -> HookJSONOutput:
        turn.retries += 1
        turn.restart_draft()
        await channel.emit(AgentMessageStream.reset())
        await channel.emit(AgentMessageStream.status("Improving the answer"))
        # Blocking the stop hands the critique back to the model, which keeps working.
        return {
            "decision": "block",
            "reason": act_feedback(decision.critique or "Improve the answer."),
        }
