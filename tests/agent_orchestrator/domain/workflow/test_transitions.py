import pytest

from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.workflow import conditions
from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.turn_policy import TurnPolicy
from tests.agent_orchestrator.domain.builders import PLAN, answer, calling, retry, turn

POLICY = TurnPolicy()


@pytest.mark.parametrize(
    ("source", "targets"),
    [
        (Step.UNDERSTAND, [Step.PLAN, Step.CLARIFY, Step.ACT]),
        (Step.ACT, [Step.PLAN, Step.FINISH, Step.RUN_TOOLS, Step.REVIEW]),
        (Step.REVIEW, [Step.FEEDBACK, Step.FINISH]),
        (Step.PLAN, [Step.ACT, Step.FINISH]),
        (Step.CLARIFY, [Step.FINISH]),
        (Step.RUN_TOOLS, [Step.ACT]),
        (Step.FEEDBACK, [Step.ACT]),
        (Step.FINISH, [Step.SUMMARIZE]),
        (Step.SUMMARIZE, [Step.END]),
    ],
)
def test_each_step_declares_where_it_can_lead(source: Step, targets: list[Step]) -> None:
    assert POLICY.targets(source) == targets


def test_every_step_but_the_end_has_a_way_out() -> None:
    assert all(POLICY.targets(step) for step in Step if step is not Step.END)


def test_nothing_leaves_the_end() -> None:
    with pytest.raises(ValueError):
        POLICY.next(Step.END, turn())


def test_conditions_read_the_turn() -> None:
    planned = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    planned.drafted(calling("echo"))
    asking = turn(Intent.DIRECT)
    asking.drafted(Draft.asking_for_plan("read it"))
    rejected = turn(Intent.CONTINUATION)
    rejected.drafted(answer())
    rejected.reviewed(retry())

    assert conditions.needs_plan(turn(Intent.TASK)) and conditions.needs_plan(turn(None))
    assert conditions.is_ambiguous(turn(Intent.AMBIGUOUS))
    assert conditions.asks_for_tools(planned)
    assert not conditions.asks_for_tools_out_of_steps(planned)
    assert conditions.asks_for_plan_without_one(asking)
    assert conditions.retry_allowed(rejected)
    assert conditions.always(turn())


def test_a_plan_approved_in_the_turn_is_carried_out_and_a_proposed_one_ends_it() -> None:
    proposed = turn(Intent.TASK)
    approved = turn(Intent.TASK, plan=PLAN)

    assert POLICY.next(Step.PLAN, proposed) == Step.FINISH
    assert POLICY.next(Step.PLAN, approved) == Step.ACT
