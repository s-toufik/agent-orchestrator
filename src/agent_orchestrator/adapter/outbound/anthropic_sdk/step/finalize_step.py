from claude_agent_sdk import ResultMessage

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState
from agent_orchestrator.adapter.outbound.anthropic_sdk.service.event_channel import EventChannel
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream

NO_ANSWER: str = "No answer was produced."
BUDGET_EXHAUSTED: str = (
    "I reached the step limit before finishing this task. "
    "Ask me to continue, or narrow the request."
)


class FinalizeStep:
    async def run(
        self, turn: TurnState, channel: EventChannel, result: ResultMessage | None = None
    ) -> tuple[str, TurnOutcome]:
        if turn.outcome is None:
            turn.answer, turn.outcome = await self._settle(turn, result, channel)
        else:
            # Plans and questions are written without the CLI: nothing was streamed yet.
            await channel.emit(AgentMessageStream.token(turn.answer or NO_ANSWER))
        answer: str = turn.answer or NO_ANSWER

        if result is not None:
            turn.agent_state.sdk_session_id = result.session_id
        turn.agent_state.record(turn.user_message, answer)
        return answer, turn.outcome

    async def _settle(
        self, turn: TurnState, result: ResultMessage | None, channel: EventChannel
    ) -> tuple[str, TurnOutcome]:
        if result is None:
            return await _replace(channel, NO_ANSWER), TurnOutcome.BEST_EFFORT

        if result.is_error:
            if not result.subtype.startswith("error_max"):
                raise RuntimeError(f"Agent run failed ({result.subtype}): {result.errors}")
            if turn.draft:
                return turn.draft, TurnOutcome.BUDGET_EXHAUSTED
            return await _replace(channel, BUDGET_EXHAUSTED), TurnOutcome.BUDGET_EXHAUSTED

        answer = turn.draft or (result.result or "").strip() or NO_ANSWER
        if not turn.draft:
            await _replace(channel, answer)
        if turn.reflection is not None and turn.reflection.should_retry:
            return answer, TurnOutcome.BEST_EFFORT
        return answer, TurnOutcome.ANSWERED


async def _replace(channel: EventChannel, answer: str) -> str:
    # Whatever was streamed is not the final answer: clear it and send the answer once.
    await channel.emit(AgentMessageStream.reset())
    await channel.emit(AgentMessageStream.token(answer))
    return answer
