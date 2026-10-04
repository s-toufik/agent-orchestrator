from typing import Protocol, runtime_checkable

from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification


@runtime_checkable
class ToolPort(Protocol):
    @property
    def specification(self) -> ToolSpecification: ...

    async def invoke(self, call: ToolCall) -> ToolResult: ...


@runtime_checkable
class ToolProviderPort(Protocol):
    async def tools(self) -> list[ToolPort]: ...
