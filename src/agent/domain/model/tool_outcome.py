from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    invocation_id: str
    tool_name: str
    output: str
    error: str | None = None

    @property
    def content(self) -> str:
        return self.output if not self.error else f"Error: {self.error}"

    @classmethod
    def failure(cls, invocation_id: str, tool_name: str, error: str) -> ToolOutcome:
        return cls(invocation_id=invocation_id, tool_name=tool_name, output="", error=error)
