from agent_orchestrator.adapter.outbound.tool.tool_registry import ToolRegistry
from agent_orchestrator.adapter.outbound.tool.toolbox import Toolbox
from agent_orchestrator.domain.tool.tool_call import ToolCall
from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.tool.tool_specification import ToolSpecification


class EchoTool:
    specification = ToolSpecification("echo", "Echoes.")

    async def invoke(self, call: ToolCall) -> ToolResult:
        return ToolResult(call.id, call.name, f"echo:{call.arguments['text']}")


class BrokenTool:
    specification = ToolSpecification("broken", "Always raises.")

    async def invoke(self, call: ToolCall) -> ToolResult:
        raise RuntimeError("boom")


def _toolbox(logger) -> Toolbox:
    return Toolbox(ToolRegistry([EchoTool(), BrokenTool()]), logger)


def test_the_catalog_lists_every_discovered_tool(logger) -> None:
    assert [tool.name for tool in _toolbox(logger).tools()] == ["echo", "broken"]


async def test_a_call_runs_its_tool(logger) -> None:
    result = await _toolbox(logger).run(ToolCall("1", "echo", {"text": "hi"}))

    assert result == ToolResult("1", "echo", "echo:hi")


async def test_an_unknown_tool_is_a_failed_result_not_an_exception(logger) -> None:
    result = await _toolbox(logger).run(ToolCall("1", "missing"))

    assert result.error and "missing" in result.error
    assert logger.messages("warning")


async def test_a_tool_that_raises_is_a_failed_result(logger) -> None:
    result = await _toolbox(logger).run(ToolCall("1", "broken"))

    assert result.error == "boom"
    assert result.content == "Error: boom"
    assert logger.messages("error")
