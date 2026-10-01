import pytest

from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from agent_orchestrator.domain.exception.unknown_tool_exception import UnknownToolException
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification


class StubTool:
    def __init__(self, name: str) -> None:
        self._specification = ToolSpecification(
            name=name, description=f"{name} tool.", parameters={"type": "object"}
        )

    @property
    def specification(self) -> ToolSpecification:
        return self._specification

    async def invoke(self, call: ToolCall) -> ToolResult:  # pragma: no cover
        return ToolResult(call_id=call.id, tool_name=call.name, output="")


def test_lookup_by_name() -> None:
    registry = ToolRegistry([StubTool("a"), StubTool("b")])

    assert registry.get("a").specification.name == "a"
    assert len(registry) == 2


def test_unknown_tool_raises_a_domain_exception() -> None:
    registry = ToolRegistry([StubTool("a")])

    with pytest.raises(UnknownToolException):
        registry.get("missing")
