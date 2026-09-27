from agent_orchestrator.domain.model.tool_specification import ToolSpecification

NO_TOOLS: str = "(no tools are available right now)"


def tool_catalog(tools: list[ToolSpecification]) -> str:
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools) or NO_TOOLS
