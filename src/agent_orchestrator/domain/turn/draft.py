from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent_orchestrator.domain.tool.tool_call import ToolCall


@dataclass(frozen=True, slots=True)
class Draft:
    text: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    plan_request: str | None = None
    kind: Literal["draft"] = "draft"

    @classmethod
    def asking_for_plan(cls, reason: str) -> Draft:
        return cls(text="", plan_request=reason)

    @property
    def asks_for_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def asks_for_plan(self) -> bool:
        return self.plan_request is not None
