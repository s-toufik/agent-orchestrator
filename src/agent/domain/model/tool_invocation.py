from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
