"""Minimal, generic MCP server scaffolding for tests.

Standing up a real MCP server here (rather than mocking the client) lets
the agent's MCP client code (StreamableHttpSessionFactory, McpToolProvider,
AgentDI, AgentContainer) be exercised against real wire traffic. This has
no dependency on any particular tool server implementation -- it's just the
`mcp` SDK's own server, wired up the same minimal way any MCP server would
be.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any
from urllib.parse import urlparse

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pycraftcore.application_configuration.model.connector import McpConnector
from starlette.applications import Starlette


def build_mcp_server(
    name: str,
    version: str,
    instructions: str | None = None,
    lifespan: Callable[[MCPServer], AbstractAsyncContextManager[Any]] | None = None,
) -> MCPServer:
    return MCPServer(name=name, version=version, instructions=instructions, lifespan=lifespan)


def build_mcp_asgi_app(server: MCPServer, connector: McpConnector) -> Starlette:
    url = urlparse(connector.base_url)
    path = url.path or "/mcp"

    return server.streamable_http_app(
        streamable_http_path=path,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[url.netloc] if url.netloc else [],
        ),
    )
