from __future__ import annotations

from dataclasses import dataclass

from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult

MAX_ARGUMENT_CHARS: int = 80
MAX_OUTCOME_CHARS: int = 300


@dataclass(frozen=True, slots=True)
class Action:
    tool: str
    arguments: str
    succeeded: bool
    outcome: str

    @classmethod
    def of(cls, call: ToolCall, result: ToolResult) -> Action:
        return cls(
            tool=call.name,
            arguments=_short_arguments(call),
            succeeded=result.error is None,
            outcome=_clip(result.content, MAX_OUTCOME_CHARS),
        )

    def render(self) -> str:
        status = "ok" if self.succeeded else "failed"
        return f"{self.tool}({self.arguments}): {status} -> {self.outcome}"


def _short_arguments(call: ToolCall) -> str:
    # Long values (code, file contents) say nothing a reader needs; names and paths do.
    return ", ".join(
        f"{name}={value!r}"
        for name, value in call.arguments.items()
        if isinstance(value, str | int | float | bool) and len(repr(value)) <= MAX_ARGUMENT_CHARS
    )


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + " [...]"
