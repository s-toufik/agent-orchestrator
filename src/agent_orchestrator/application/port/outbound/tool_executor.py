from typing import Protocol

from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult


class ToolExecutor(Protocol):
    async def run(self, call: ToolCall) -> ToolResult: ...
