import pytest

from agent.adapter.outbound.langgraph.service.tool_mapper import to_langchain_tools
from agent.adapter.outbound.tool.tool_registry import ToolRegistry
from agent.domain.exception.unknown_tool_exception import UnknownToolException
from agent.domain.model.tool_invocation import ToolInvocation
from agent.domain.model.tool_outcome import ToolOutcome
from agent.domain.model.tool_specification import ToolSpecification


class StubTool:
    def __init__(self, name: str) -> None:
        self._specification = ToolSpecification(
            name=name, description=f"{name} tool.", parameters={"type": "object"}
        )

    @property
    def specification(self) -> ToolSpecification:
        return self._specification

    async def invoke(self, invocation: ToolInvocation) -> ToolOutcome:  # pragma: no cover
        return ToolOutcome(invocation_id=invocation.id, tool_name=invocation.name, output="")


def test_lookup_by_name() -> None:
    registry = ToolRegistry([StubTool("a"), StubTool("b")])

    assert registry.get("a").specification.name == "a"
    assert len(registry) == 2


def test_unknown_tool_raises_a_domain_exception() -> None:
    registry = ToolRegistry([StubTool("a")])

    with pytest.raises(UnknownToolException):
        registry.get("missing")


def test_specifications_map_to_the_bind_tools_shape() -> None:
    registry = ToolRegistry([StubTool("a")])

    bound = to_langchain_tools(registry.specifications())

    assert bound == [{"name": "a", "description": "a tool.", "parameters": {"type": "object"}}]
