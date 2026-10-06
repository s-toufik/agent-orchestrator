from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.turn.action import Action
from agent_orchestrator.domain.turn.answer import BUDGET_EXHAUSTED, NO_ANSWER, Answer
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.outcome import Outcome
from tests.agent_orchestrator.domain.builders import (
    PLAN,
    accept,
    answer,
    calling,
    failure,
    result,
    retry,
    turn,
)


def test_an_accepted_draft_is_the_answer() -> None:
    current = turn(Intent.CONTINUATION)
    current.drafted(answer("42"))
    current.reviewed(accept())

    assert current.conclude() == Answer.answered("42")


def test_a_draft_still_rejected_is_a_best_effort_answer() -> None:
    current = turn(Intent.CONTINUATION)
    current.drafted(answer("41"))
    current.reviewed(retry())

    assert current.conclude() == Answer("41", Outcome.BEST_EFFORT)


def test_pending_tool_calls_after_the_plan_mean_the_step_budget_ran_out() -> None:
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.drafted(calling("echo"))
    current.observed([result()])
    current.drafted(calling("echo"))

    assert current.conclude() == Answer(BUDGET_EXHAUSTED, Outcome.BUDGET_EXHAUSTED)


def test_an_unfinished_plan_is_reported_step_by_step_not_claimed_as_done() -> None:
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.drafted(answer("Step 1: I counted the rows."))

    concluded = current.conclude()

    assert concluded.outcome is Outcome.BUDGET_EXHAUSTED
    assert "→ 1. count (tool: echo)" in concluded.text
    assert "I counted" not in concluded.text


def test_the_progress_follows_the_successful_results_of_the_plan() -> None:
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    assert current.leaves_plan_unfinished

    current.observed([result()])

    assert current.progress is not None and current.progress.is_complete
    assert not current.leaves_plan_unfinished
    assert turn(Intent.DIRECT).progress is None


def test_actions_pair_each_tool_call_with_its_result() -> None:
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.drafted(calling("echo", "file_writer"))
    current.observed([result(output="hi", call_id="c0"), failure("file_writer", call_id="c1")])

    assert current.actions == [
        Action("echo", "", True, "hi"),
        Action("file_writer", "", False, "Error: boom"),
    ]


def test_an_action_keeps_short_arguments_and_clips_the_outcome() -> None:
    call = ToolCall("1", "file_writer", {"file_path": "r.md", "data": "x" * 500, "n": 2})

    action = Action.of(call, result("file_writer", "line one\n" + "y" * 400))

    assert action.arguments == "file_path='r.md', n=2"
    assert action.render().startswith("file_writer(file_path='r.md', n=2): ok -> line one yyy")
    assert action.outcome.endswith(" [...]") and len(action.outcome) == 306


def test_no_draft_means_no_answer() -> None:
    assert turn().conclude() == Answer(NO_ANSWER, Outcome.BEST_EFFORT)


def test_steps_retries_and_evidence_are_read_from_the_work_done() -> None:
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    current.drafted(calling("echo"))
    current.observed([result(output="hi")])
    current.drafted(answer())
    current.give_feedback("shorter")
    current.drafted(answer("short"))

    assert current.steps_taken == 3
    assert current.retries == 1
    assert [item.output for item in current.evidence] == ["hi"]
    assert current.last_draft == answer("short")
    assert current.used_tools


def test_only_a_direct_answer_without_tools_is_plain() -> None:
    plain = turn(Intent.DIRECT)
    with_tools = turn(Intent.DIRECT)
    with_tools.observed([result()])

    assert plain.is_plain_direct_answer
    assert not with_tools.is_plain_direct_answer
    assert not turn(Intent.CONTINUATION).is_plain_direct_answer


def test_the_query_is_the_understood_request_or_the_raw_message() -> None:
    assert turn(Intent.DIRECT).query == "the query"
    assert turn(None).query == "msg"


def test_the_approval_request_is_the_plan_awaiting_approval() -> None:
    approval = Answer.approval_request(PLAN)

    assert (approval.text, approval.outcome) == (PLAN.render(), Outcome.AWAITING_APPROVAL)
    assert approval.text == "## Plan\n1. count (tool: echo)\n## Expected result\nA count."


def test_a_plan_request_is_neither_an_answer_nor_tool_calls() -> None:
    request = Draft.asking_for_plan("read the report")

    assert request.asks_for_plan and request.plan_request == "read the report"
    assert not request.asks_for_tools
    assert not answer().asks_for_plan


def test_carrying_out_a_plan_starts_from_fresh_work() -> None:
    current = turn(Intent.DIRECT)
    current.drafted(Draft.asking_for_plan("read the report"))
    current.reviewed(accept())

    current.execute(PLAN)

    assert current.plan == PLAN
    assert current.work == [] and current.verdicts == []
