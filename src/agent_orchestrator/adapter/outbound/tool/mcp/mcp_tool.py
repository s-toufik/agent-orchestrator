import json
import traceback
from typing import Any

from mcp.types import CallToolResult, TextContent

from agent_orchestrator.adapter.outbound.tool.mcp.mcp_session_factory import McpSessionFactory
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification


class McpTool:
    def __init__(
        self, session_factory: McpSessionFactory, specification: ToolSpecification
    ) -> None:
        self._session_factory = session_factory
        self._specification = specification

    @property
    def specification(self) -> ToolSpecification:
        return self._specification

    async def invoke(self, call: ToolCall) -> ToolResult:
        try:
            async with self._session_factory.session() as session:
                result: CallToolResult = await session.call_tool(
                    self._specification.name, call.arguments
                )
        except Exception as exception:
            traceback_str: str = "".join(traceback.format_exception(exception))
            return ToolResult.failure(
                call_id=call.id,
                tool_name=self._specification.name,
                error=f"MCP call failed: {traceback_str}",
            )

        return self._to_result(call, result)

    def _to_result(self, call: ToolCall, result: Any) -> ToolResult:
        if not isinstance(result, CallToolResult):
            return ToolResult.failure(
                call_id=call.id,
                tool_name=self._specification.name,
                error=f"Unsupported MCP result type: {type(result).__name__}",
            )

        if result.is_error:
            return ToolResult.failure(
                call_id=call.id,
                tool_name=self._specification.name,
                error=self._stringify(result.content),
            )
        return ToolResult(
            call_id=call.id,
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
