from collections.abc import Sequence

from agent_orchestrator.domain.workflow.step import Step
from agent_orchestrator.domain.workflow.turn_policy import TurnPolicy

Hop = tuple[Step, Step]


class RouteCoverage:
    def __init__(self, policy: TurnPolicy | None = None) -> None:
        transitions = (policy or TurnPolicy()).TRANSITIONS
        self._known: list[Hop] = list(dict.fromkeys((t.source, t.target) for t in transitions))
        self._seen: set[Hop] = set()

    def record(self, route: Sequence[Step]) -> None:
        self._seen.update(zip(route, (*route[1:], Step.END), strict=True))

    @property
    def is_empty(self) -> bool:
        return not self._seen

    def missing(self) -> list[Hop]:
        return [hop for hop in self._known if hop not in self._seen]

    def report(self) -> list[str]:
        covered = len(self._known) - len(self.missing())
        lines = [f"{covered}/{len(self._known)} transitions taken"]
        lines += [f"  not taken: {source} -> {target}" for source, target in self.missing()]
        return lines
