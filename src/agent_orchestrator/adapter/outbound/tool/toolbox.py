from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from agent_orchestrator.domain.exception.unknown_tool_exception import UnknownToolException
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification


class Toolbox:
    def __init__(self, registry: ToolRegistry, logger: Logger) -> None:
        self._registry = registry
        self._logger = logger

    def tools(self) -> list[ToolSpecification]:
        return self._registry.specifications()

    async def run(self, call: ToolCall) -> ToolResult:
        try:
            return await self._registry.get(call.name).invoke(call)
        except UnknownToolException as exception:
            self._logger.warning(f"[{call.id}] {exception}")
            return ToolResult.failure(call.id, call.name, str(exception))
        except Exception as exception:
            self._logger.error(f"[{call.id}] tool '{call.name}' raised: {exception}")
            return ToolResult.failure(call.id, call.name, str(exception))
