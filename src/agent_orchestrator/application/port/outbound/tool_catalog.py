from typing import Protocol

from agent_orchestrator.domain.tool.tool_specification import ToolSpecification


class ToolCatalog(Protocol):
    def tools(self) -> list[ToolSpecification]: ...
