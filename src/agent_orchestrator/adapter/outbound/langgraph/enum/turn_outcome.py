from enum import StrEnum


class TurnOutcome(StrEnum):
    ANSWERED = "answered"
    BEST_EFFORT = "best_effort"
    BUDGET_EXHAUSTED = "budget_exhausted"
    AWAITING_APPROVAL = "awaiting_approval"
    CLARIFICATION = "clarification"
