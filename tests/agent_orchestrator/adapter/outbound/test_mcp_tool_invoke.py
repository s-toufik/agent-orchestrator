"""Exercises McpTool.invoke against a real MCP server (see tests/support/mcp_test_server.py
for why: this is real wire traffic, not a mocked ClientSession).
"""

import asyncio
import socket
from collections.abc import AsyncIterator

import pytest
import uvicorn
from mcp.server.mcpserver.exceptions import ToolError
from pycraftcore.application_configuration.enum import ConnectorType
from pycraftcore.application_configuration.model.connector import McpConnector
from pycraftcore.authentication.model.no_auth import NoAuth
from pydantic import BaseModel

from agent_orchestrator.adapter.outbound.tool.mcp.mcp_tool import McpTool
from agent_orchestrator.adapter.outbound.tool.mcp.streamable_http_session_factory import (
    StreamableHttpSessionFactory,
)
from agent_orchestrator.domain.model.tool_invocation import ToolInvocation
from agent_orchestrator.domain.model.tool_specification import ToolSpecification
from tests.support.mcp_test_server import build_mcp_asgi_app, build_mcp_server


class Rows(BaseModel):
    rows: list[dict[str, int]]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def running_server() -> AsyncIterator[str]:
    server = build_mcp_server(name="toolbox", version="1.0.0")

    @server.tool(name="structured", description="Returns structured content.")
    async def structured() -> Rows:
        return Rows(rows=[{"n": 1}])

    @server.tool(
        name="unstructured", description="Returns plain text only.", structured_output=False
    )
    async def unstructured() -> str:
        return "plain text"

    @server.tool(name="always_fails", description="Always fails.")
    async def always_fails() -> str:
        raise ToolError("boom")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/mcp"
    connector = McpConnector(
        name="self",
        type=ConnectorType.mcp,
        auth=NoAuth(),
        base_url=base_url,
        timeout=5,
        transport="streamable_http",
    )
    app = build_mcp_asgi_app(server, connector)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv_server = uvicorn.Server(config)
    serve_task = asyncio.create_task(uv_server.serve())
    try:
        while not uv_server.started:
            await asyncio.sleep(0.01)
        yield base_url
    finally:
        uv_server.should_exit = True
        await serve_task


def _tool(base_url: str, name: str, logger) -> McpTool:
    connector = McpConnector(
        name="self",
        type=ConnectorType.mcp,
        auth=NoAuth(),
        base_url=base_url,
        timeout=5,
        transport="streamable_http",
    )
    session_factory = StreamableHttpSessionFactory(connector, logger)
    return McpTool(session_factory, ToolSpecification(name=name, description="."))


async def test_prefers_structured_content_when_present(running_server, logger) -> None:
    tool = _tool(running_server, "structured", logger)

    outcome = await tool.invoke(ToolInvocation(id="1", name="structured"))

    assert not outcome.error
    assert outcome.output == '{"rows": [{"n": 1}]}'


async def test_falls_back_to_text_when_no_structured_content(running_server, logger) -> None:
    tool = _tool(running_server, "unstructured", logger)

    outcome = await tool.invoke(ToolInvocation(id="1", name="unstructured"))

    assert not outcome.error
    assert outcome.output == "plain text"


async def test_failure_still_reads_the_text_error_message(running_server, logger) -> None:
    tool = _tool(running_server, "always_fails", logger)

    outcome = await tool.invoke(ToolInvocation(id="1", name="always_fails"))

    assert outcome.error
    assert "boom" in outcome.error
