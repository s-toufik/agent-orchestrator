from mcp import ListToolsResult

from agent_orchestrator.adapter.outbound.tool.mcp.mcp_session_factory import McpSessionFactory
from agent_orchestrator.adapter.outbound.tool.mcp.mcp_tool import McpTool
from agent_orchestrator.adapter.outbound.tool.tool_port import ToolPort
from agent_orchestrator.domain.exception.tool_unavailable_exception import ToolUnavailableException
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification


class McpToolProvider:
    def __init__(self, session_factory: McpSessionFactory, required: bool = True) -> None:
        self._session_factory = session_factory
        self._required = required

    async def tools(self) -> list[ToolPort]:
        async with self._session_factory.session() as session:
            response: ListToolsResult = await session.list_tools()

        tools: list[ToolPort] = [
            McpTool(
                session_factory=self._session_factory,
                specification=ToolSpecification(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.input_schema,
                    output_schema=tool.output_schema,
                ),
            )
            for tool in response.tools
        ]

        if self._required and not tools:
            raise ToolUnavailableException("The MCP server advertised no tool")

        return tools
