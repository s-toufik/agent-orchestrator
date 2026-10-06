from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.workflow.step import Step
from tests.evaluation.harness.exchange import Exchange, names
from tests.evaluation.harness.score import Check

ENDING: tuple[Step, ...] = (Step.FINISH, Step.SUMMARIZE)


def contract_checks(exchange: Exchange) -> list[Check]:
    turn, route = exchange.turn, exchange.route
    checks = [
        Check(
            "answered",
            exchange.answer is not None and bool(exchange.answer.text.strip()),
            "the turn ended without an answer",
        ),
        Check(
            "route shape",
            route[:1] == (Step.UNDERSTAND,) and route[-2:] == ENDING,
            f"the route {names(route)} does not run understand ... finish, summarize",
        ),
        Check(
            "tools under a plan",
            not exchange.tools or turn.plan is not None,
            f"tools {list(exchange.tools)} ran without an approved plan",
        ),
        Check(
            "step budget",
            turn.steps_taken <= turn.settings.max_steps,
            f"{turn.steps_taken} steps taken, over the budget of {turn.settings.max_steps}",
        ),
    ]
    if exchange.outcome is Outcome.AWAITING_APPROVAL:
        checks.append(
            Check(
                "plan pending",
                exchange.conversation.pending_plan is not None,
                "the answer awaits approval but no plan is pending",
            )
        )
    if exchange.outcome is Outcome.ANSWERED and turn.plan is not None:
        checks += [
            Check(
                "plan finished",
                not turn.leaves_plan_unfinished,
                "the plan is answered as done but steps are still open",
            ),
            Check(
                "plan reviewed",
                Step.REVIEW in route,
                "the answer to a plan was not reviewed",
            ),
        ]
    return checks
