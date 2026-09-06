from agent.application.port.outbound.tool_port import ToolPort
from agent.domain.exception.unknown_tool_exception import UnknownToolException
from agent.domain.model.tool_specification import ToolSpecification


class ToolRegistry:
    def __init__(self, tools: list[ToolPort]) -> None:
        self._tools: dict[str, ToolPort] = {tool.specification.name: tool for tool in tools}

    def get(self, name: str) -> ToolPort:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolException(f"Unknown tool: '{name}'.")
        return tool

    def specifications(self) -> list[ToolSpecification]:
        return [tool.specification for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
