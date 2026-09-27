from dataclasses import dataclass, field
from pathlib import Path

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.mcp_server import McpServer


@dataclass(frozen=True, slots=True)
class SdkSettings:
    base_url: str
    auth_token: str
    working_directory: Path
    builtin_tools: tuple[str, ...] = ()
    mcp_servers: list[McpServer] = field(default_factory=list)
    otlp_endpoint: str | None = None
    environment: str = ""
