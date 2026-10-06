import json
from datetime import datetime

import pytest

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step
from tests.agent_orchestrator.domain.builders import PLAN, answer, calling, result, turn
from tests.evaluation.harness.cases import CASES, PROPOSED
from tests.evaluation.harness.contract import contract_checks
from tests.evaluation.harness.exchange import Exchange
from tests.evaluation.harness.expectation import Expect
from tests.evaluation.harness.report import EvaluationReport
from tests.evaluation.harness.route_coverage import RouteCoverage
from tests.evaluation.harness.score import Check, Score, ScoreKind, threshold_from_env

DIRECT = (Step.UNDERSTAND, Step.ACT, Step.FINISH, Step.SUMMARIZE)
EXECUTED = (
    Step.UNDERSTAND,
    Step.ACT,
    Step.RUN_TOOLS,
    Step.ACT,
    Step.REVIEW,
    Step.FINISH,
    Step.SUMMARIZE,
)


def _exchange(current: Turn, route: tuple[Step, ...], pending: bool = False) -> Exchange:
    conversation = Conversation("c1")
    if pending:
        conversation.propose(PLAN)
    return Exchange("msg", route, current, conversation)


def _answered(text: str = "Paris") -> Turn:
    current = turn(Intent.DIRECT)
    current.drafted(answer(text))
    current.finish(Answer.answered(text))
    return current


def _executed(*work, outcome: Answer | None = None) -> Turn:
    current = turn(Intent.PLAN_APPROVAL, plan=PLAN)
    for item in work:
        current.drafted(item) if hasattr(item, "tool_calls") else current.observed([item])
    current.finish(outcome or current.conclude())
    return current


def _failed(checks: list[Check]) -> list[str]:
    return [check.detail for check in checks if not check.passed]


def test_an_exchange_that_meets_the_expectation_passes_every_check() -> None:
    expect = Expect(
        intents=(Intent.DIRECT,),
        outcome=Outcome.ANSWERED,
        route=DIRECT,
        tools=(),
        answer_contains=("paris",),
        plan_pending=False,
    )

    checks = expect.checks(_exchange(_answered(), DIRECT))

    assert len(checks) == 6 and _failed(checks) == []


def test_only_the_fields_set_are_checked() -> None:
    assert Expect().checks(_exchange(_answered(), DIRECT)) == []


def test_each_unmet_expectation_is_a_failed_check_with_a_readable_detail() -> None:
    expect = Expect(
        intents=(Intent.TASK,),
        outcome=Outcome.AWAITING_APPROVAL,
        route=PROPOSED,
        tools=("echo",),
        plan_tools=frozenset({"echo"}),
        plan_contains=("count",),
        answer_contains=("Rome",),
        plan_pending=True,
    )

    checks = expect.checks(_exchange(_answered(), DIRECT))

    assert [check.name for check in checks if not check.passed] == [
        "intent",
        "outcome",
        "route",
        "tools",
        "plan pending",
        "plan tools",
        "plan mentions",
        "answer mentions",
    ]
    assert "intent direct, expected one of ['task']" in _failed(checks)


def test_the_plan_is_read_from_the_turn_or_from_the_pending_plan() -> None:
    expect = Expect(plan_tools=frozenset({"echo"}), plan_contains=("count",))
    proposed = turn(Intent.TASK)
    proposed.finish(Answer.approval_request(PLAN))
    executed = _executed(calling("echo"), result(), answer())

    assert _failed(expect.checks(_exchange(proposed, PROPOSED, pending=True))) == []
    assert _failed(expect.checks(_exchange(executed, EXECUTED))) == []


def test_a_well_behaved_turn_passes_every_rule_that_applies_to_it() -> None:
    direct = contract_checks(_exchange(_answered(), DIRECT))
    executed = contract_checks(_exchange(_executed(calling("echo"), result(), answer()), EXECUTED))

    assert len(direct) == 4 and _failed(direct) == []
    assert len(executed) == 6 and _failed(executed) == []


def test_answering_a_plan_with_open_steps_fails_the_contract() -> None:
    claimed = _executed(answer("I did it all."), outcome=Answer.answered("I did it all."))

    assert _failed(contract_checks(_exchange(claimed, DIRECT))) == [
        "the plan is answered as done but steps are still open",
        "the answer to a plan was not reviewed",
    ]


def test_tools_without_a_plan_and_a_missing_answer_fail_the_contract() -> None:
    rogue = turn(Intent.DIRECT)
    rogue.drafted(calling("echo"))
    rogue.observed([result()])

    assert _failed(contract_checks(_exchange(rogue, (Step.UNDERSTAND, Step.ACT)))) == [
        "the turn ended without an answer",
        "the route ['understand', 'act'] does not run understand ... finish, summarize",
        "tools ['echo'] ran without an approved plan",
    ]


def test_awaiting_approval_without_a_pending_plan_fails_the_contract() -> None:
    proposed = turn(Intent.TASK)
    proposed.finish(Answer.approval_request(PLAN))

    assert _failed(contract_checks(_exchange(proposed, PROPOSED))) == [
        "the answer awaits approval but no plan is pending"
    ]


def test_a_route_score_is_the_share_of_checks_passed() -> None:
    checks = [Check("a", True), Check("b", True), Check("c", True), Check("d", False, "d failed")]

    score = Score.of_checks("case", checks, threshold=0.7)

    assert (score.value, score.kind, score.failures) == (0.75, ScoreKind.ROUTE, ("d failed",))
    assert score.passed
    assert not Score.of_checks("case", checks, threshold=0.8).passed
    assert score.explain() == "case: 0.75, at or above the threshold 0.70\n- d failed"


def test_the_threshold_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_THRESHOLD", raising=False)
    assert threshold_from_env() == 0.8

    monkeypatch.setenv("EVAL_THRESHOLD", "0.65")
    assert threshold_from_env() == 0.65

    for wrong in ("high", "1.5"):
        monkeypatch.setenv("EVAL_THRESHOLD", wrong)
        with pytest.raises(ValueError, match="EVAL_THRESHOLD"):
            threshold_from_env()


def test_the_report_is_written_as_markdown_and_json(tmp_path) -> None:
    coverage = RouteCoverage()
    coverage.record(DIRECT)
    report = EvaluationReport("nvidia/model-x", 0.8, coverage)
    report.add(Score("good case", ScoreKind.ROUTE, 1.0, 0.8))
    report.add(Score("weak case", ScoreKind.QUALITY, 0.5, 0.8, ("Bias: one-sided",)))
    report.add_unscored("crashed case", "AgentUnavailableException: 429 Too Many Requests")

    markdown, data = report.write(tmp_path / "reports", datetime(2026, 10, 6, 14, 30))

    assert markdown.name == "20261006-143000_nvidia-model-x.md"
    text = markdown.read_text()
    assert "mean score **0.75** · **1/2** at or above the threshold · **1** not scored" in text
    assert "- **crashed case**: AgentUnavailableException: 429 Too Many Requests" in text
    assert text.index("| 0.50 | FAIL | quality | weak case |") < text.index("| 1.00 | pass |")
    assert "### weak case (0.50)\n\n- Bias: one-sided" in text
    assert "4/17 transitions taken" in text
    saved = json.loads(data.read_text())
    assert (saved["model"], saved["mean_score"], saved["passed"], saved["total"]) == (
        "nvidia/model-x",
        0.75,
        1,
        2,
    )
    assert saved["scores"][1] == {
        "name": "weak case",
        "kind": "quality",
        "score": 0.5,
        "passed": False,
        "failures": ["Bias: one-sided"],
    }
    assert saved["unscored"] == [
        {"name": "crashed case", "error": "AgentUnavailableException: 429 Too Many Requests"}
    ]
    assert report.summary()[-1] == "1 not scored: they failed to run"


def test_route_coverage_lists_the_transitions_never_taken() -> None:
    coverage = RouteCoverage()
    assert coverage.is_empty

    coverage.record(DIRECT)
    coverage.record(EXECUTED)

    missing = coverage.missing()
    assert (Step.UNDERSTAND, Step.ACT) not in missing and (Step.SUMMARIZE, Step.END) not in missing
    assert (Step.UNDERSTAND, Step.CLARIFY) in missing and (Step.ACT, Step.ACT) in missing
    assert coverage.report()[0] == f"{17 - len(missing)}/17 transitions taken"
    assert "  not taken: understand -> clarify" in coverage.report()


def test_every_case_is_named_once_and_says_something() -> None:
    names = [case.name for case in CASES]

    assert len(names) == len(set(names))
    assert all(case.says for case in CASES)
