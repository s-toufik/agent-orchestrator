from contextlib import AbstractAsyncContextManager
from typing import Any

from agent_orchestrator.adapter.outbound.tool.mcp.mcp_tool import McpTool
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification

SPEC = ToolSpecification(name="run_sql", description="SQL.", parameters={"type": "object"})


class _UnusedSessionFactory:
    def session(self) -> AbstractAsyncContextManager[Any]:
        raise NotImplementedError


def test_specification_property_exposes_the_tool_specification() -> None:
    tool = McpTool(_UnusedSessionFactory(), SPEC)

    assert tool.specification is SPEC
