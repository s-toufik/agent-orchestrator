from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.turn.plan import Plan, PlannedStep

DONE, CURRENT, TO_DO = "✓", "→", "·"


@dataclass(frozen=True, slots=True)
class PlanProgress:
    plan: Plan
    done: int

    @classmethod
    def of(cls, plan: Plan, results: Sequence[ToolResult]) -> PlanProgress:
        progress = cls(plan, 0).past_reasoning()
        for result in results:
            if progress.is_done_by(result):
                progress = cls(plan, progress.done + 1).past_reasoning()
        return progress

    @property
    def current(self) -> PlannedStep | None:
        return None if self.is_complete else self.plan.steps[self.done]

    @property
    def number(self) -> int:
        return self.done + 1

    @property
    def is_complete(self) -> bool:
        return self.done >= len(self.plan.steps)

    def is_done_by(self, result: ToolResult) -> bool:
        step = self.current
        return step is not None and result.error is None and result.tool_name == step.tool

    def past_reasoning(self) -> PlanProgress:
        done = self.done
        while done < len(self.plan.steps) and self.plan.steps[done].tool is None:
            done += 1
        return PlanProgress(self.plan, done)

    def render(self) -> str:
        return "\n".join(
            f"{self._mark(index)} {index + 1}. {step.render()}"
            for index, step in enumerate(self.plan.steps)
        )

    def _mark(self, index: int) -> str:
        if index < self.done:
            return DONE
        return CURRENT if index == self.done else TO_DO
