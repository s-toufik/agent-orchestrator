from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ToolAccess:
    tools: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
