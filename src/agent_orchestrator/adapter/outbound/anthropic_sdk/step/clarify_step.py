from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_state import TurnState


class ClarifyStep:
    # The context step already wrote the question: no model call, as in the LangGraph engine.
    @staticmethod
    def run(turn: TurnState) -> None:
        turn.answer = turn.context.clarification_question
        turn.outcome = TurnOutcome.CLARIFICATION
