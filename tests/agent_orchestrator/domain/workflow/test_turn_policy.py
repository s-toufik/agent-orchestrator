import pytest

from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.turn_policy import TurnPolicy
from tests.agent_orchestrator.domain.builders import (
    PLAN,
    accept,
    answer,
    calling,
    result,
    retry,
    turn,
)

POLICY = TurnPolicy()


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (Intent.TASK, Step.PLAN),
        (Intent.PLAN_REVISION, Step.PLAN),
        (None, Step.PLAN),
        (Intent.AMBIGUOUS, Step.CLARIFY),
        (Intent.DIRECT, Step.ACT),
        (Intent.CONTINUATION, Step.ACT),
        (Intent.PLAN_APPROVAL, Step.ACT),
    ],
)
def test_after_understanding_the_intent_picks_the_route(
    intent: Intent | None, expected: Step
) -> None:
    assert POLICY.next(Step.UNDERSTAND, turn(intent)) == expected


@pytest.mark.parametrize("done", [Step.PLAN, Step.CLARIFY])
def test_a_plan_or_a_question_ends_the_work(done: Step) -> None:
    assert POLICY.next(done, turn(Intent.TASK)) == Step.FINISH


def _acted(current: Turn, *work) -> Turn:
    for item in work:
        if hasattr(item, "tool_calls"):
            current.drafted(item)
        else:
            current.observed([item])
    return current


def test_tool_calls_are_run_while_steps_remain() -> None:
    current = _acted(turn(Intent.PLAN_APPROVAL, plan=PLAN), calling("echo"))

    assert POLICY.next(Step.ACT, current) == Step.RUN_TOOLS


def test_tool_calls_after_the_last_step_end_the_turn() -> None:
    current = _acted(
        turn(Intent.PLAN_APPROVAL, max_steps=2), calling("echo"), result(), calling("echo")
    )

    assert POLICY.next(Step.ACT, current) == Step.FINISH


def test_a_plain_direct_answer_is_not_reviewed() -> None:
    assert POLICY.next(Step.ACT, _acted(turn(Intent.DIRECT), answer())) == Step.FINISH


@pytest.mark.parametrize(
    "current",
    [
        _acted(turn(Intent.CONTINUATION), answer()),
        _acted(turn(Intent.DIRECT), calling("echo"), result(), answer()),
        _acted(turn(Intent.PLAN_APPROVAL, plan=PLAN), answer()),
    ],
)
def test_any_other_answer_is_reviewed(current: Turn) -> None:
    assert POLICY.next(Step.ACT, current) == Step.REVIEW


@pytest.mark.parametrize("done", [Step.RUN_TOOLS, Step.FEEDBACK])
def test_tools_and_feedback_lead_back_to_acting(done: Step) -> None:
    assert POLICY.next(done, turn()) == Step.ACT


def _reviewed(verdict, retries: int = 0, steps: int = 1, max_steps: int = 5) -> Turn:
    current = turn(Intent.CONTINUATION, max_steps=max_steps)
    for _ in range(steps):
        current.drafted(answer())
    for _ in range(retries):
        current.give_feedback("fix it")
    current.reviewed(verdict)
    return current


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (_reviewed(retry()), Step.FEEDBACK),
        (_reviewed(accept()), Step.FINISH),
        (_reviewed(retry(), retries=2), Step.FINISH),
        (_reviewed(retry(), steps=5), Step.FINISH),
    ],
)
def test_a_rejected_answer_is_retried_only_within_both_limits(
    current: Turn, expected: Step
) -> None:
    assert POLICY.next(Step.REVIEW, current) == expected


def test_a_finished_turn_is_summarized_then_ends() -> None:
    assert POLICY.next(Step.FINISH, turn()) == Step.SUMMARIZE
    assert POLICY.next(Step.SUMMARIZE, turn()) == Step.END


def test_an_actor_asking_for_a_plan_gets_one_when_the_turn_has_none() -> None:
    current = _acted(turn(Intent.DIRECT), Draft.asking_for_plan("read the report"))

    assert POLICY.next(Step.ACT, current) == Step.PLAN


def test_under_an_approved_plan_a_plan_request_is_just_a_draft_to_review() -> None:
    current = _acted(turn(Intent.PLAN_APPROVAL, plan=PLAN), Draft.asking_for_plan("more"))

    assert POLICY.next(Step.ACT, current) == Step.REVIEW
