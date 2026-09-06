from typing import Any

from agent.domain.model.tool_specification import ToolSpecification


def to_langchain_tool(specification: ToolSpecification) -> dict[str, Any]:

    return {
        "name": specification.name,
        "description": specification.description,
        "parameters": specification.parameters,
    }


def to_langchain_tools(specifications: list[ToolSpecification]) -> list[dict[str, Any]]:
    return [to_langchain_tool(specification) for specification in specifications]
