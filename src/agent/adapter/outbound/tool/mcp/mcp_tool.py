from typing import Any

from mcp.types import CallToolResult, TextContent

from agent.adapter.outbound.tool.mcp.mcp_session_factory import McpSessionFactory
from agent.domain.model.tool_invocation import ToolInvocation
from agent.domain.model.tool_outcome import ToolOutcome
from agent.domain.model.tool_specification import ToolSpecification


class McpTool:
    def __init__(
        self, session_factory: McpSessionFactory, specification: ToolSpecification
    ) -> None:
        self._session_factory = session_factory
        self._specification = specification

    @property
    def specification(self) -> ToolSpecification:
        return self._specification

    async def invoke(self, invocation: ToolInvocation) -> ToolOutcome:
        try:
            session = await self._session_factory.session()
            result = await session.call_tool(self._specification.name, invocation.arguments)
        except Exception as exception:
            await self._session_factory.invalidate()
            return ToolOutcome.failure(
                invocation_id=invocation.id,
                tool_name=self._specification.name,
                error=f"MCP call failed: {exception}",
            )

        return self._to_outcome(invocation, result)

    def _to_outcome(self, invocation: ToolInvocation, result: Any) -> ToolOutcome:
        if not isinstance(result, CallToolResult):
            return ToolOutcome.failure(
                invocation_id=invocation.id,
                tool_name=self._specification.name,
                error=f"Unsupported MCP result type: {type(result).__name__}",
            )

        output = self._stringify(result.content)
        if result.is_error:
            return ToolOutcome.failure(
                invocation_id=invocation.id, tool_name=self._specification.name, error=output
            )
        return ToolOutcome(
            invocation_id=invocation.id, tool_name=self._specification.name, output=output
        )

    @staticmethod
    def _stringify(content: list[Any]) -> str:
        return "\n".join(
            block.text if isinstance(block, TextContent) else str(block) for block in content
        )
