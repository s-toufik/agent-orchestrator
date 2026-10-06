from __future__ import annotations

from dataclasses import dataclass

from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.plan_progress import PlanProgress

NO_ANSWER: str = "No answer was produced."
BUDGET_EXHAUSTED: str = (
    "I reached the step limit before finishing this task. "
    "Ask me to continue, or narrow the request."
)
PLAN_UNFINISHED: str = (
    "I reached the step limit before finishing the plan:\n\n{progress}\n\n"
    "Ask me to continue, or narrow the request."
)
NO_PLAN: str = "I could not write a plan for this request. Try rephrasing it."


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
    def plan_unfinished(cls, progress: PlanProgress) -> Answer:
        return cls(PLAN_UNFINISHED.format(progress=progress.render()), Outcome.BUDGET_EXHAUSTED)

    @classmethod
    def approval_request(cls, plan: Plan) -> Answer:
        return cls(plan.render(), Outcome.AWAITING_APPROVAL)

    @classmethod
    def no_plan(cls) -> Answer:
        return cls(NO_PLAN, Outcome.BEST_EFFORT)

    @classmethod
    def clarification(cls, question: str) -> Answer:
        return cls(question, Outcome.CLARIFICATION)
