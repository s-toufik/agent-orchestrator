from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class McpServer:
    name: str
    url: str
    transport: Literal["http", "sse"] = "http"
    headers: dict[str, str] = field(default_factory=dict)
