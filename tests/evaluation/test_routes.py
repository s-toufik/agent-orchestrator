import pytest

from tests.evaluation.harness.cases import CASES, Case
from tests.evaluation.harness.contract import contract_checks
from tests.evaluation.harness.score import Check, Score


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
async def test_the_agent_takes_the_expected_route(
    case: Case, new_session, threshold, scorecard
) -> None:
    session = new_session()
    checks: list[Check] = []
    for number, say in enumerate(case.says, 1):
        exchange = await session.say(say.message, auto_approve=say.auto_approve)
        where = f"message {number} {say.message!r}"
        checks += [
            Check(check.name, check.passed, f"{where}: {check.detail}")
            for check in say.expect.checks(exchange) + contract_checks(exchange)
        ]

    scorecard(Score.of_checks(case.name, checks, threshold))
