import asyncio
import socket
from collections.abc import AsyncIterator
from typing import Any

import pytest
import uvicorn
from mcp.types import TextContent
from pycraftcore.application_configuration.enum import ConnectorType
from pycraftcore.application_configuration.model.connector import McpConnector
from pycraftcore.authentication.model.no_auth import NoAuth

from agent.adapter.outbound.tool.mcp.streamable_http_session_factory import (
    StreamableHttpSessionFactory,
)
from agent.domain.exception.tool_unavailable_exception import ToolUnavailableException
from tests.support.mcp_test_server import build_mcp_asgi_app, build_mcp_server


def _text(result: Any) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def running_toolbox() -> AsyncIterator[str]:
    """A real toolbox MCP server, listening on a real socket, with one tool."""
    server = build_mcp_server(name="toolbox", version="1.0.0")

    @server.tool(name="echo", description="Echoes its input back.")
    async def echo(text: str) -> str:
        return text

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/mcp"
    connector = McpConnector(
        name="self",
        type=ConnectorType.mcp,
        auth=NoAuth(),
        base_url=base_url,
        timeout=30,
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


def _client_connector(base_url: str) -> McpConnector:
    return McpConnector(
        name="toolbox",
        type=ConnectorType.mcp,
        auth=NoAuth(),
        base_url=base_url,
        timeout=5,
        transport="streamable_http",
    )


async def test_connects_over_a_real_http_transport_and_calls_a_real_tool(
    running_toolbox, logger
) -> None:
    factory = StreamableHttpSessionFactory(
        connector=_client_connector(running_toolbox),
        logger=logger,
        max_attempts=3,
        initial_delay=0.05,
        max_delay=0.05,
    )

    await factory.start()
    session = await factory.session()
    result = await session.call_tool("echo", {"text": "hi"})

    assert _text(result) == "hi"

    await factory.close()


async def test_invalidate_then_session_reconnects_for_real(running_toolbox, logger) -> None:
    factory = StreamableHttpSessionFactory(
        connector=_client_connector(running_toolbox),
        logger=logger,
        max_attempts=3,
        initial_delay=0.05,
        max_delay=0.05,
    )

    await factory.start()
    first = await factory.session()
    await factory.invalidate()
    second = await factory.session()

    assert first is not second
    result = await second.call_tool("echo", {"text": "again"})
    assert _text(result) == "again"

    await factory.close()


async def test_a_transport_that_connects_but_speaks_the_wrong_protocol_fails_cleanly(
    running_toolbox, logger
) -> None:
    # The TCP connection succeeds against a real, reachable server -- just not
    # an MCP one at this path -- so _connect()'s cleanup path runs against an
    # already-healthy transport instead of a broken one.
    non_mcp_url = running_toolbox.rsplit("/mcp", 1)[0] + "/health"
    factory = StreamableHttpSessionFactory(
        connector=_client_connector(non_mcp_url),
        logger=logger,
        max_attempts=1,
        initial_delay=0.01,
        max_delay=0.01,
    )

    with pytest.raises(ToolUnavailableException):
        await factory.start()


async def test_sse_transport_branch_is_exercised(running_toolbox, logger) -> None:
    # The toolbox only serves streamable-http, so this can't succeed -- the
    # point is exercising _open_transport()'s "sse" branch at all.
    connector = McpConnector(
        name="toolbox",
        type=ConnectorType.mcp,
        auth=NoAuth(),
        base_url=running_toolbox,
        timeout=1,
        transport="sse",
    )
    factory = StreamableHttpSessionFactory(
        connector=connector, logger=logger, max_attempts=1, initial_delay=0.01, max_delay=0.01
    )

    with pytest.raises(ToolUnavailableException):
        await factory.start()


async def test_gives_up_for_real_when_nothing_ever_listens(logger) -> None:
    # Deterministic counterpart to the "eventually connects" test below: this
    # port never gets a listener, so every real attempt genuinely fails --
    # exercising _connect()'s and _open_transport()'s actual failure paths
    # rather than the monkeypatched _connect used for the pure retry-logic tests.
    port = _free_port()
    factory = StreamableHttpSessionFactory(
        connector=_client_connector(f"http://127.0.0.1:{port}/mcp"),
        logger=logger,
        max_attempts=2,
        initial_delay=0.01,
        max_delay=0.01,
    )

    with pytest.raises(ToolUnavailableException):
        await factory.start()

    assert len(logger.messages("warning")) == 2


async def test_start_retries_until_a_server_appears(logger) -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/mcp"
    factory = StreamableHttpSessionFactory(
        connector=_client_connector(base_url),
        logger=logger,
        max_attempts=30,
        initial_delay=0.05,
        max_delay=0.05,
    )

    server = build_mcp_server(name="toolbox", version="1.0.0")
    connector = McpConnector(
        name="self",
        type=ConnectorType.mcp,
        auth=NoAuth(),
        base_url=base_url,
        timeout=30,
        transport="streamable_http",
    )
    app = build_mcp_asgi_app(server, connector)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv_server = uvicorn.Server(config)

    start_task = asyncio.create_task(factory.start())
    await asyncio.sleep(0.2)  # let a couple of failed attempts happen against nothing listening yet
    assert not start_task.done()

    serve_task = asyncio.create_task(uv_server.serve())
    try:
        while not uv_server.started:
            await asyncio.sleep(0.01)
        await start_task
        assert len(logger.messages("warning")) >= 1
    finally:
        uv_server.should_exit = True
        await serve_task
        await factory.close()
