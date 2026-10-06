from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlannedStep:
    action: str
    tool: str | None = None

    def render(self) -> str:
        return f"{self.action} (tool: {self.tool or 'none'})"


@dataclass(frozen=True, slots=True)
class Plan:
    task: str
    steps: tuple[PlannedStep, ...]
    expected_result: str = ""

    def render(self) -> str:
        lines = ["## Plan", *(f"{n}. {step.render()}" for n, step in enumerate(self.steps, 1))]
        if self.expected_result:
            lines += ["## Expected result", self.expected_result]
        return "\n".join(lines)
