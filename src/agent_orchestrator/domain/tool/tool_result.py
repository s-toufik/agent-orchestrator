from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    output: str
    error: str | None = None
    kind: Literal["tool_result"] = "tool_result"

    @property
    def content(self) -> str:
        return f"Error: {self.error}" if self.error else self.output

    @classmethod
    def failure(cls, call_id: str, tool_name: str, error: str) -> ToolResult:
        return cls(call_id=call_id, tool_name=tool_name, output="", error=error)
