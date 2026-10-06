from collections.abc import Callable

from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.turn import Turn

Condition = Callable[[Turn], bool]


def always(turn: Turn) -> bool:
    return True


def needs_plan(turn: Turn) -> bool:
    return turn.intent is None or turn.intent.needs_plan


def has_approved_plan(turn: Turn) -> bool:
    return turn.plan is not None


def is_ambiguous(turn: Turn) -> bool:
    return turn.intent is Intent.AMBIGUOUS


def asks_for_plan_without_one(turn: Turn) -> bool:
    draft = turn.last_draft
    return draft is not None and draft.asks_for_plan and turn.plan is None


def asks_for_tools(turn: Turn) -> bool:
    draft = turn.last_draft
    return draft is not None and draft.asks_for_tools


def leaves_plan_unfinished(turn: Turn) -> bool:
    return turn.last_draft is not None and turn.leaves_plan_unfinished


def keeps_working_out_of_steps(turn: Turn) -> bool:
    keeps_working = asks_for_tools(turn) or leaves_plan_unfinished(turn)
    return keeps_working and turn.steps_taken >= turn.settings.max_steps


def is_plain_direct_answer(turn: Turn) -> bool:
    return turn.is_plain_direct_answer


def retry_allowed(turn: Turn) -> bool:
    verdict = turn.last_verdict
    return (
        verdict is not None
        and verdict.rejects
        and turn.retries < turn.settings.max_retries
        and turn.steps_taken < turn.settings.max_steps
    )
