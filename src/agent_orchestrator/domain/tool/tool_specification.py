from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSpecification:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
