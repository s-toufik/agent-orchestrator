from __future__ import annotations

from dataclasses import dataclass

from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.turn.plan import Plan

NO_ANSWER: str = "No answer was produced."
BUDGET_EXHAUSTED: str = (
    "I reached the step limit before finishing this task. "
    "Ask me to continue, or narrow the request."
)
APPROVAL_REQUEST: str = "{steps}\n\n---\nReply **yes** to run this plan, or tell me what to change."


@dataclass(frozen=True, slots=True)
class Answer:

    text: str
    outcome: Outcome

    @classmethod
    def answered(cls, text: str) -> Answer:
        return cls(text, Outcome.ANSWERED)

    @classmethod
    def best_effort(cls, text: str) -> Answer:
        return cls(text, Outcome.BEST_EFFORT)

    @classmethod
    def nothing(cls) -> Answer:
        return cls(NO_ANSWER, Outcome.BEST_EFFORT)

    @classmethod
    def budget_exhausted(cls) -> Answer:
        return cls(BUDGET_EXHAUSTED, Outcome.BUDGET_EXHAUSTED)

    @classmethod
    def approval_request(cls, plan: Plan) -> Answer:
        return cls(APPROVAL_REQUEST.format(steps=plan.steps), Outcome.AWAITING_APPROVAL)

    @classmethod
    def clarification(cls, question: str) -> Answer:
        return cls(question, Outcome.CLARIFICATION)
