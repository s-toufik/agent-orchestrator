from typing import Protocol, runtime_checkable

from agent.domain.model.tool_invocation import ToolInvocation
from agent.domain.model.tool_outcome import ToolOutcome
from agent.domain.model.tool_specification import ToolSpecification


@runtime_checkable
class ToolPort(Protocol):
    @property
    def specification(self) -> ToolSpecification: ...

    async def invoke(self, invocation: ToolInvocation) -> ToolOutcome: ...


@runtime_checkable
class ToolRegistryPort(Protocol):
    def get(self, name: str) -> ToolPort: ...

    def specifications(self) -> list[ToolSpecification]: ...


@runtime_checkable
class ToolProviderPort(Protocol):
    async def tools(self) -> list[ToolPort]: ...
