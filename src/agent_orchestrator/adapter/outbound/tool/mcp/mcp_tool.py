import json
import traceback
from typing import Any

from mcp.types import CallToolResult, TextContent

from agent_orchestrator.adapter.outbound.tool.mcp.mcp_session_factory import McpSessionFactory
from agent_orchestrator.domain.model.tool_invocation import ToolInvocation
from agent_orchestrator.domain.model.tool_outcome import ToolOutcome
from agent_orchestrator.domain.model.tool_specification import ToolSpecification


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
            async with self._session_factory.session() as session:
                result: CallToolResult = await session.call_tool(
                    self._specification.name, invocation.arguments
                )
        except Exception as exception:
            traceback_str: str = "".join(traceback.format_exception(exception))
            return ToolOutcome.failure(
                invocation_id=invocation.id,
                tool_name=self._specification.name,
                error=f"MCP call failed: {traceback_str}",
            )

        return self._to_outcome(invocation, result)

    def _to_outcome(self, invocation: ToolInvocation, result: Any) -> ToolOutcome:
        if not isinstance(result, CallToolResult):
            return ToolOutcome.failure(
                invocation_id=invocation.id,
                tool_name=self._specification.name,
                error=f"Unsupported MCP result type: {type(result).__name__}",
            )

        if result.is_error:
            return ToolOutcome.failure(
                invocation_id=invocation.id,
                tool_name=self._specification.name,
                error=self._stringify(result.content),
            )
        return ToolOutcome(
            invocation_id=invocation.id,
            tool_name=self._specification.name,
            output=self._render_output(result),
        )

    @classmethod
    def _render_output(cls, result: CallToolResult) -> str:
        if result.structured_content is not None:
            try:
                return json.dumps(result.structured_content)
            except TypeError, ValueError:
                pass
        return cls._stringify(result.content)

    @staticmethod
    def _stringify(content: list[Any]) -> str:
        return "\n".join(
            block.text if isinstance(block, TextContent) else str(block) for block in content
        )
