from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.prompt.reflection_prompt import (
    reflection_request,
    reflection_system_prompt,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.model_profile import ModelProfile
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.structured_output import (
    StructuredOutput,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream


class ReflectStep:
    def __init__(
        self,
        structured_output: StructuredOutput,
        max_evidence_chars: int = 2_000,
        max_message_chars: int = 4_000,
    ) -> None:
        self._structured_output = structured_output
        self._max_evidence_chars = max_evidence_chars
        self._max_message_chars = max_message_chars

    @staticmethod
    def applies(turn: TurnState) -> bool:
        # Plans and questions are not answers; a direct answer without tools has no evidence.
        if turn.mode in (TurnMode.PLAN, TurnMode.CLARIFY):
            return False
        return not (turn.context.intent is Intent.DIRECT and not turn.tools_used)

    async def run(
        self, profile: ModelProfile, turn: TurnState, channel: EventChannel
    ) -> ReflectionDecision | None:
        await channel.emit(AgentMessageStream.status("Checking the answer"))
        context = turn.context
        return await self._structured_output.invoke_structured(
            profile=profile,
            system=reflection_system_prompt(ReflectionDecision.model_json_schema()),
            prompt=reflection_request(
                conversation=self._recent_conversation(turn),
                request=context.standalone_query,
                criteria=context.success_criteria,
                plan=turn.plan.steps if turn.plan else None,
                evidence=[self._clip(item, self._max_evidence_chars) for item in turn.evidence],
                answer=turn.draft,
            ),
            schema=ReflectionDecision,
        )

    def _recent_conversation(self, turn: TurnState) -> str:
        lines: list[str] = []
        previous = turn.agent_state.last_exchange()
        if previous is not None:
            lines.append(f"User: {self._clip(previous.user, self._max_message_chars)}")
            lines.append(f"Assistant: {self._clip(previous.assistant, self._max_message_chars)}")
        lines.append(f"User: {self._clip(turn.user_message, self._max_message_chars)}")
        return "\n".join(lines)

    @staticmethod
    def _clip(content: str, limit: int) -> str:
        return content if len(content) <= limit else content[:limit] + " [...]"
