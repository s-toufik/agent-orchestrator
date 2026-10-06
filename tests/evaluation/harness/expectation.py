from dataclasses import dataclass, field

from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.workflow.step import Step
from tests.evaluation.harness.exchange import Exchange, names
from tests.evaluation.harness.score import Check


@dataclass(frozen=True, slots=True)
class Expect:
    intents: tuple[Intent, ...] = ()
    outcome: Outcome | None = None
    route: tuple[Step, ...] | None = None
    tools: tuple[str, ...] | None = None
    plan_tools: frozenset[str] = field(default_factory=frozenset)
    plan_contains: tuple[str, ...] = ()
    answer_contains: tuple[str, ...] = ()
    plan_pending: bool | None = None

    def checks(self, exchange: Exchange) -> list[Check]:
        found: list[Check] = []
        if self.intents:
            expected = [str(intent) for intent in self.intents]
            found.append(
                Check(
                    "intent",
                    exchange.intent in self.intents,
                    f"intent {exchange.intent}, expected one of {expected}",
                )
            )
        if self.outcome is not None:
            found.append(
                Check(
                    "outcome",
                    exchange.outcome is self.outcome,
                    f"outcome {exchange.outcome}, expected {self.outcome}",
                )
            )
        if self.route is not None:
            found.append(
                Check(
                    "route",
                    exchange.route == self.route,
                    f"route {names(exchange.route)}, expected {names(self.route)}",
                )
            )
        if self.tools is not None:
            found.append(
                Check(
                    "tools",
                    exchange.tools == self.tools,
                    f"tools run {list(exchange.tools)}, expected {list(self.tools)}",
                )
            )
        return found + self._plan_checks(exchange) + self._answer_checks(exchange)

    def _plan_checks(self, exchange: Exchange) -> list[Check]:
        found: list[Check] = []
        pending = exchange.conversation.pending_plan is not None
        if self.plan_pending is not None:
            found.append(
                Check(
                    "plan pending",
                    pending is self.plan_pending,
                    f"plan pending is {pending}, expected {self.plan_pending}",
                )
            )
        plan = exchange.plan
        if self.plan_tools:
            planned = {step.tool for step in plan.steps if step.tool} if plan else set()
            found.append(
                Check(
                    "plan tools",
                    self.plan_tools <= planned,
                    f"plan uses {sorted(planned)}, expected {sorted(self.plan_tools)}",
                )
            )
        rendered = plan.render().lower() if plan else ""
        found += [
            Check("plan mentions", part.lower() in rendered, f"plan does not mention {part!r}")
            for part in self.plan_contains
        ]
        return found

    def _answer_checks(self, exchange: Exchange) -> list[Check]:
        text = exchange.answer.text.lower() if exchange.answer else ""
        return [
            Check("answer mentions", part.lower() in text, f"answer does not mention {part!r}")
            for part in self.answer_contains
        ]
